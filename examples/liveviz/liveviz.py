from charm4py import charm, Chare, Array, Future, Reducer, Group, liveviz, coro
import random

class Unit(Chare):
  
  def __init__(self):
    self.colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    
  def reqImg(self, request):
    self.particles = []
    
    for _ in range(300):
      x = random.randint(0, 49)
      y = random.randint(0, 49)
      
      color = random.choice(self.colors)
      
      self.particles.append((x, y, color))
    
    data = bytearray(50 * 50 * 3)
    
    for x, y, (r, g, b) in self.particles:
      pixel_index = (y * 50 + x) * 3
      data[pixel_index] = r
      data[pixel_index + 1] = g
      data[pixel_index + 2] = b
    
    liveviz.LiveViz.deposit(data, self, self.thisIndex[0]*50, self.thisIndex[1]*50, 50, 50, 800, 800)

def main(args):
    units = Array(Unit, dims=(16,16))
    config = liveviz.Config()
    liveviz.LiveViz.init(config, units.reqImg)
    print("CCS Handlers registered . Waiting for net requests...")

charm.start(main)
