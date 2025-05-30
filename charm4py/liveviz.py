from .charm import charm, Array
from dataclasses import dataclass, field
import struct

@dataclass
class Config:
	version: int = 1
	isColor: bool = True
	isPush: bool = True
	is3d: bool = False
	min: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))
	max: tuple = field(default_factory=lambda: (1.0, 1.0, 1.0))
	
	def to_binary(self):
		# Format binary data compatible with Java's DataInputStream
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
		
		# Unpack the fixed fields
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
		
class ByteImage:
    def __init__(self, data, width, height, is_color=True):
        """
        Initialize a byte image
        
        Args:
            data (bytes): Raw image data as bytes
            width (int): Image width in pixels
            height (int): Image height in pixels 
            is_color (bool): Whether the image is in color (True) or grayscale (False)
        """
        self.data = data
        self.width = width
        self.height = height
        self.is_color = is_color
    
    def to_binary(self):
        # Format: width (int), height (int), isColor (int), followed by raw bytes
        header = struct.pack(">iii", 
                           self.width, 
                           self.height,
                           1 if self.is_color else 0)
        return header + self.data

class LiveViz:
	def __init__(self):
		self.cfg = None
		self.arr = None
		self.callback = None
			
	def config_handler(self, msg):
		charm.CcsSendReply(self.cfg.to_binary())
			
	def image_handler(self, msg):
		request = ImageRequest.from_bytes(msg)
		self.callback(request)
			
	def init(self, cfg, arr, cb):
		self.cfg = cfg
		self.arr = arr
		self.callback = cb
		charm.CcsRegisterHandler("lvConfig", self.config_handler)
		charm.CcsRegisterHandler("lvImage", self.image_handler)