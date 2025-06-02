from .charm import charm, Chare
from dataclasses import dataclass, field
import struct
from itertools import chain
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
  def __init__(self, version, request_type, width, height, 
              x=None, y=None, z=None, o=None, minZ=0.0, maxZ=0.0):
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
  
  @classmethod
  def from_bytes(cls, data):
    if len(data) < 16:  # At least 4 ints
      raise ValueError("Not enough data to decode ImageRequest")
    
    version, request_type, width, height = struct.unpack_from(">iiii", data, 0)
    
    # If there's more data, we have the optional fields
    if len(data) > 16:
      offset = 16
      x, offset = Vector3d.from_bytes(data, offset)
      y, offset = Vector3d.from_bytes(data, offset)
      z, offset = Vector3d.from_bytes(data, offset)
      o, offset = Vector3d.from_bytes(data, offset)
      minZ, maxZ = struct.unpack_from(">dd", data, offset)
      
      return cls(version, request_type, width, height, x, y, z, o, minZ, maxZ)
    else:
      return cls(version, request_type, width, height)
  
class LiveVizGroup(Chare):
  
  def __init__(self, cb):
    self.callback = cb
    charm.CcsRegisterHandler("lvImage", self.image_handler)

  def send(self, result):
    image = ByteImage.from_contributions(result, LiveViz.cfg.isColor)
    output = ByteImage.with_image_in_corner(image, self.wid, self.ht)
    charm.CcsSendDelayedReply(self.reply, output.to_binary())

  def image_handler(self, msg):
    request = ImageRequest.from_bytes(msg)
    self.ht = request.height
    self.wid = request.width
    self.callback(request)
    self.reply = charm.CcsDelayReply()
  
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
  
  @classmethod
  def from_contributions(cls, contribs, is_color=True):
    """
    Create a ByteImage from multiple contributions, positioning each
    contribution at the right location.
    
    Args:
        contribs (list): List of tuples with format 
            (bytes_data, startx, starty, local_height, local_width, total_height, total_width)
        is_color (bool): Whether the image is in color
    
    Returns:
        ByteImage: A composite image with all contributions in the right positions
    """        
    _, _, _, _, _, total_height, total_width = contribs[0]
    bytes_per_pixel = 3 if is_color else 1
    
    buffer = bytearray(total_width * total_height * bytes_per_pixel)
    
    for data, startx, starty, local_height, local_width, _, _ in contribs:
      for y in range(local_height):
        for x in range(local_width):
          src_pos = (y * local_width + x) * bytes_per_pixel
          dst_pos = ((starty + y) * total_width + (startx + x)) * bytes_per_pixel
          
          if src_pos + bytes_per_pixel <= len(data):
            buffer[dst_pos:dst_pos + bytes_per_pixel] = data[src_pos:src_pos + bytes_per_pixel]
    
    return cls(bytes(buffer), total_width, total_height, is_color)
  
  def to_binary(self):
    return self.data

  @classmethod
  def with_image_in_corner(cls, src_image, new_width, new_height):
    """
    Create a new image with specified dimensions and place the source image
    in the top left corner.
    
    Args:
        src_image (ByteImage): Source image to place in the corner
        new_width (int): Width of the new image
        new_height (int): Height of the new image
        
    Returns:
        ByteImage: A new image with the source image in the top left corner
    """
    dest_image = cls(None, new_width, new_height, src_image.is_color)
    bytes_per_pixel = dest_image.bytes_per_pixel
    
    buffer = bytearray(new_width * new_height * bytes_per_pixel)
    
    # Calculate dimensions to copy
    copy_width = min(new_width, src_image.width)
    copy_height = min(new_height, src_image.height)
    
    for y in range(copy_height):
      for x in range(copy_width):
        src_pos = (y * src_image.width + x) * bytes_per_pixel
        
        dst_pos = (y * new_width + x) * bytes_per_pixel
        
        if src_pos + bytes_per_pixel <= len(src_image.data):
          buffer[dst_pos:dst_pos + bytes_per_pixel] = src_image.data[src_pos:src_pos + bytes_per_pixel]
    
    return cls(bytes(buffer), new_width, new_height, src_image.is_color)

class LiveViz:
  cfg = None
  
  @classmethod
  def config_handler(cls, msg):
    charm.CcsSendReply(cls.cfg.to_binary())
  
  @classmethod
  def deposit(cls, buffer, elem, x, y, ht, wid, g_ht, g_wid):
    elem.reduce(group.send, data=(buffer,x,y,ht,wid,g_ht,g_wid), reducer=Reducer.viz_gather)
  
  @classmethod
  def init(cls, cfg, arr, cb):
    global group
    cls.cfg = cfg
    grp = Chare(LiveVizGroup, args=[cb], onPE=0)
    charm.thisProxy.updateGlobals({'group': grp}, awaitable=True, module_name='charm4py.liveviz').get()
    charm.CcsRegisterHandler("lvConfig", cls.config_handler)
