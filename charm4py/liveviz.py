from .charm import charm, Chare, register
from dataclasses import dataclass, field
from collections import deque
import io
import struct
import sys
from itertools import chain
import numpy as np
Reducer = charm.reducers

group = None

def viz_gather(contribs):
    return list(chain(*contribs))

def viz_gather_preprocess(data, contributor):
    return [data]

Reducer.addReducer(viz_gather, pre=viz_gather_preprocess)

@dataclass
class Config:
  version: int = 1
  isColor: bool = True
  isPush: bool = True
  is3d: bool = False
  min: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))
  max: tuple = field(default_factory=lambda: (1.0, 1.0, 1.0))
  
  def to_binary(self):
    # Format: int, int, int, int, [double, double, double, double, double, double]
    binary_data = struct.pack(">iiii", 
                            self.version,
                            1 if self.isColor else 0,
                            1 if self.isPush else 0,
                            1 if self.is3d else 0)
    if self.is3d:
      binary_data += struct.pack(">dddddd",
                              self.min[0], self.min[1], self.min[2],
                              self.max[0], self.max[1], self.max[2])
    return binary_data
    
# liveVizRequest::compressionType values (charm/src/libs/ck-libs/liveViz/liveViz0.h)
COMPRESSION_NONE = 0
COMPRESSION_JPEG = 1

_warned = set()

def _warn_once(key, message):
  if key not in _warned:
    _warned.add(key)
    print('charm4py.liveviz WARNING: ' + message, file=sys.stderr)

class Vector3d:
  def __init__(self, x=0.0, y=0.0, z=0.0):
    self.x = x
    self.y = y
    self.z = z

  @classmethod
  def from_bytes(cls, data, offset=0):
    # Read 3 doubles from the data starting at offset
    x, y, z = struct.unpack_from(">ddd", data, offset)
    return cls(x, y, z), offset + 24  # 24 = 3 * 8 bytes (double)

class ImageRequest:
  """An lvImage request, laid out as liveVizRequest::pupNetwork (liveViz0.C):
  int version, code, width, height; if version >= 2 also int compressionType
  and compressionQuality; then, from 3D clients, the view (vectors x, y, z, o
  and doubles minZ, maxZ). All big-endian."""

  HEADER_V1 = struct.Struct(">iiii")
  HEADER_V2_EXTRA = struct.Struct(">ii")
  VIEW_3D_SIZE = 4 * 24 + 2 * 8

  def __init__(self, version, request_type, width, height,
              x=None, y=None, z=None, o=None, minZ=0.0, maxZ=0.0,
              compression_type=COMPRESSION_NONE, compression_quality=0):
    self.version = version
    self.request_type = request_type
    self.width = width
    self.height = height
    self.x = x
    self.y = y
    self.z = z
    self.o = o
    self.minZ = minZ
    self.maxZ = maxZ
    self.compression_type = compression_type
    self.compression_quality = compression_quality

  @classmethod
  def from_bytes(cls, data):
    if len(data) < cls.HEADER_V1.size:  # At least 4 ints
      raise ValueError("Not enough data to decode ImageRequest")

    version, request_type, width, height = cls.HEADER_V1.unpack_from(data, 0)
    offset = cls.HEADER_V1.size
    compression_type, compression_quality = COMPRESSION_NONE, 0
    if version >= 2:  # newer clients also send the compression fields
      if len(data) < offset + cls.HEADER_V2_EXTRA.size:
        raise ValueError("Not enough data to decode version %d ImageRequest" % version)
      compression_type, compression_quality = cls.HEADER_V2_EXTRA.unpack_from(data, offset)
      offset += cls.HEADER_V2_EXTRA.size

    req = cls(version, request_type, width, height,
              compression_type=compression_type,
              compression_quality=compression_quality)
    # 3D clients append their view; anything else after the header is ignored
    if len(data) - offset >= cls.VIEW_3D_SIZE:
      req.x, offset = Vector3d.from_bytes(data, offset)
      req.y, offset = Vector3d.from_bytes(data, offset)
      req.z, offset = Vector3d.from_bytes(data, offset)
      req.o, offset = Vector3d.from_bytes(data, offset)
      req.minZ, req.maxZ = struct.unpack_from(">dd", data, offset)
    return req

_pil_image = None

def _pillow():
  """Pillow's Image module (imported and its JPEG encoder loaded once), or
  None when Pillow is not installed."""
  global _pil_image
  if _pil_image is None:
    try:
      from PIL import Image, JpegImagePlugin  # noqa: F401
      _pil_image = Image
    except ImportError:
      _pil_image = False
  return _pil_image or None

def encode_reply(image, request):
  """The lvImage reply for an image already sized to the request, as
  liveViz0Deposit does it: raw pixels, a JPEG stream, or nothing for an
  unsupported compression type."""
  ctype = request.compression_type
  if ctype == COMPRESSION_NONE:
    return image.to_binary()
  if ctype == COMPRESSION_JPEG:
    Image = _pillow()
    if Image is None:
      _warn_once('jpeg', 'JPEG image requested but Pillow is not installed; '
                 'sending empty replies (install Pillow to enable JPEG)')
      return b''
    mode = 'RGB' if image.is_color else 'L'
    # libjpeg's jpeg_set_quality clamps the quality to 1..100 the same way
    quality = min(max(request.compression_quality, 1), 100)
    try:
      pil = Image.frombuffer(mode, (image.width, image.height), image.to_binary(),
                             'raw', mode, 0, 1)
      out = io.BytesIO()
      pil.save(out, format='JPEG', quality=quality)
    except Exception as e:
      _warn_once(('jpegfail', str(e)), 'JPEG compression of a %dx%d image failed '
                 '(%s); sending an empty reply' % (image.width, image.height, e))
      return b''
    return out.getvalue()
  _warn_once(('ctype', ctype), 'ignoring liveViz client\'s unrecognized '
             'compressionType %d (sending empty replies)' % ctype)
  return b''

@register
class LiveVizGroup(Chare):

  def __init__(self, cb, poll):
    self.callback = cb
    self.poll = poll
    charm.CcsRegisterHandler("lvImage", self.image_handler)
    # Importing Pillow can take seconds on a cold parallel file system: do it
    # now rather than while the first JPEG client waits.
    _pillow()
    # (request, delayed reply token) waiting for an image, oldest first
    self.requests = deque()
    if poll:
      self.images = deque()

  @staticmethod
  def _reply_bytes(image, request):
    output = ByteImage.with_image_in_corner(image, request.width, request.height)
    return encode_reply(output, request)

  def send(self, result):
    image = ByteImage.from_contributions(result, LiveViz.cfg.isColor)
    if len(self.requests) > 0:
      # Array reductions complete in the order they were started, so in push
      # mode the oldest pending request is the one this image answers.
      req, delayed = self.requests.popleft()
      charm.CcsSendDelayedReply(delayed, self._reply_bytes(image, req))
    elif self.poll:
      self.images.append(image)
    else:
      _warn_once('unrequested', 'push-mode image arrived with no pending '
                 'request; dropping it')

  def image_handler(self, msg):
    request = ImageRequest.from_bytes(msg)
    if self.poll:
      if len(self.images) > 0:
        charm.CcsSendReply(self._reply_bytes(self.images.popleft(), request))
      else:
        self.requests.append((request, charm.CcsDelayReply()))
    else:
      # Delay the reply before asking for the image, as liveViz0 does, and
      # queue it so overlapping requests each get their own image.
      self.requests.append((request, charm.CcsDelayReply()))
      self.callback(request)

class ByteImage:
  def __init__(self, data=None, width=0, height=0, is_color=True):
    """
    Initialize a byte image

    Args:
        data (bytes, optional): Raw image data as bytes, or None to create empty image
        width (int): Image width in pixels
        height (int): Image height in pixels
        is_color (bool): Whether the image is in color (True) or grayscale (False)
    """
    self.width = width
    self.height = height
    self.is_color = is_color
    self.bytes_per_pixel = 3 if is_color else 1

    if data is not None:
      self.data = data
    else:
      self.data = bytes(width * height * self.bytes_per_pixel)

  @staticmethod
  def _pixels(data, width, height, bytes_per_pixel):
    """data as a (height, width, bytes_per_pixel) uint8 array. Missing
    trailing bytes are zero (as if those pixels were not drawn); extra
    bytes are ignored."""
    if isinstance(data, np.ndarray):
      flat = data.reshape(-1).astype(np.uint8, copy=False)
    else:
      flat = np.frombuffer(data, dtype=np.uint8)
    n = width * height * bytes_per_pixel
    if flat.size < n:
      flat = np.concatenate((flat, np.zeros(n - flat.size, dtype=np.uint8)))
    return flat[:n].reshape(height, width, bytes_per_pixel)

  def pixels(self):
    return self._pixels(self.data, self.width, self.height, self.bytes_per_pixel)

  @classmethod
  def from_contributions(cls, contribs, is_color=True):
    """
    Create a ByteImage from multiple contributions, positioning each
    contribution at the right location. Overlapping pixel values are added
    modulo 256; parts of a contribution outside the image are clipped.

    Args:
        contribs (list): List of tuples with format
            (bytes_data, startx, starty, local_height, local_width, total_height, total_width)
        is_color (bool): Whether the image is in color

    Returns:
        ByteImage: A composite image with all contributions in the right positions
    """
    _, _, _, _, _, total_height, total_width = contribs[0]
    bytes_per_pixel = 3 if is_color else 1

    buffer = np.zeros((total_height, total_width, bytes_per_pixel), dtype=np.uint8)

    for data, startx, starty, local_height, local_width, _, _ in contribs:
      if local_width <= 0 or local_height <= 0:
        continue
      # Intersect [startx, startx+local_width) x [starty, ...) with the image
      x0, y0 = max(startx, 0), max(starty, 0)
      x1 = min(startx + local_width, total_width)
      y1 = min(starty + local_height, total_height)
      if x0 >= x1 or y0 >= y1:
        continue
      src = cls._pixels(data, local_width, local_height, bytes_per_pixel)
      # uint8 addition wraps, i.e. adds modulo 256
      buffer[y0:y1, x0:x1] += src[y0 - starty:y1 - starty, x0 - startx:x1 - startx]

    return cls(buffer.tobytes(), total_width, total_height, is_color)

  def to_binary(self):
    if isinstance(self.data, bytes):
      return self.data
    return bytes(self.data)

  @classmethod
  def with_image_in_corner(cls, src_image, new_width, new_height):
    """
    Create a new image with specified dimensions and place the source image
    in the top left corner (cropped if it is larger, padded with black if
    it is smaller).

    Args:
        src_image (ByteImage): Source image to place in the corner
        new_width (int): Width of the new image
        new_height (int): Height of the new image

    Returns:
        ByteImage: A new image with the source image in the top left corner
    """
    new_width, new_height = max(new_width, 0), max(new_height, 0)
    if (new_width == src_image.width and new_height == src_image.height
        and isinstance(src_image.data, bytes)
        and len(src_image.data) == new_width * new_height * src_image.bytes_per_pixel):
      return cls(src_image.data, new_width, new_height, src_image.is_color)

    bytes_per_pixel = src_image.bytes_per_pixel
    buffer = np.zeros((new_height, new_width, bytes_per_pixel), dtype=np.uint8)

    # Calculate dimensions to copy
    copy_width = min(new_width, src_image.width)
    copy_height = min(new_height, src_image.height)
    if copy_width > 0 and copy_height > 0:
      buffer[:copy_height, :copy_width] = src_image.pixels()[:copy_height, :copy_width]

    return cls(buffer.tobytes(), new_width, new_height, src_image.is_color)

class LiveViz:
  cfg = None
  
  @classmethod
  def config_handler(cls, msg):
    charm.CcsSendReply(cls.cfg.to_binary())
  
  @classmethod
  def deposit(cls, buffer, elem, x, y, ht, wid, g_ht, g_wid):
    elem.reduce(group.send, data=(buffer,x,y,ht,wid,g_ht,g_wid), reducer=Reducer.viz_gather)
  
  @classmethod
  def init(cls, cfg, cb, poll=False):
    global group
    cls.cfg = cfg
    grp = Chare(LiveVizGroup, args=[cb, poll], onPE=0)
    charm.thisProxy.updateGlobals({'group': grp}, awaitable=True, module_name='charm4py.liveviz').get()
    charm.CcsRegisterHandler("lvConfig", cls.config_handler)
