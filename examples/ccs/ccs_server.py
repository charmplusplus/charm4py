from charm4py import charm, Chare, Array, Future, Reducer, Group, liveviz, coro
import random

class Unit(Chare):
  
  def __init__(self):
    # Just initialize an empty list - particles will be generated on request
    self.colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    
  def reqImg(self, request):
    # Generate 300 new random particles each time
    self.particles = []
    
    for _ in range(300):
      # Random position within 50x50 box
      x = random.randint(0, 49)
      y = random.randint(0, 49)
      
      # Random color from the three options
      color = random.choice(self.colors)
      
      # Store particle as (position_x, position_y, color_tuple)
      self.particles.append((x, y, color))
    
    # Create and fill the image data
    data = bytearray(50 * 50 * 3)
    
    for x, y, (r, g, b) in self.particles:
      # Check if particle is in this chare's section
      pixel_index = (y * 50 + x) * 3
      data[pixel_index] = r
      data[pixel_index + 1] = g
      data[pixel_index + 2] = b
    
    liveviz.LiveViz.deposit(data, self, self.thisIndex[0]*50, self.thisIndex[1]*50, 50, 50, 800, 800)

def main(args):
    # No need to initialize converse, because charm.start does this
    # just register the handler
    reg_wait = Future()
    units = Array(Unit, dims=(16,16))
    config = liveviz.Config()
    liveviz.LiveViz.init(config, units, units.reqImg)
    print("CCS Handlers registered . Waiting for net requests...")

charm.start(main, modules=['charm4py.liveviz'])