import array
import random
import numba
import math
import numpy as np
import time
from charm4py import *

class GlobalDefs:
    # These need to be member variables because it simplifies broadcasting
    def __init__( self ):
        self.BLOCK_SIZE = 512
        self.HYDROGEN_MASS = ( 1.67 * 1e-24 ) # in g
        self.VDW_A = ( 1.1328 * 1e-133 )  # in (g m^2/s^2) m^12
        self.VDW_B = ( 2.23224 * 1e-76 ) # (g m^2/s^2) m^6

        self.ENERGY_VAR = (1.0 * 1e-5 )

        # average of next two should be what you want as your atom density
        # this should comply with the PERDIM parameter; for KAWAY 1 1 1, the maximum number
        # of particles can be 10*10*10 = 1000 - 10 comes from PERDIM parameter, which is
        # currently set to be 10, using a GAP of 3; as KWAYness increases, the maximum
        # number of particles decreases - for 2 1 1, it is 500, for 2 2 1 it is 250; you
        # can set them to have lower values but not higher; alternatively a host of
        # paramters including PTP_CUT_OFF, PERDIM, GAP can be set to suitable values to
        self.PARTICLES_PER_CELL_START = 100
        self.PARTICLES_PER_CELL_END = 250

        self.DEFAULT_DELTA = 1 # in femtoseconds

        self.DEFAULT_FIRST_LDB = 20
        self.DEFAULT_LDB_PERIOD = 20
        self.DEFAULT_FT_PERIOD = 100000

        self.KAWAY_X = 2
        self.KAWAY_Y = 2
        self.KAWAY_Z = 1
        self.NBRS_X  = (2*self.KAWAY_X+1)
        self.NBRS_Y = (2*self.KAWAY_Y+1)
        self.NBRS_Z = (2*self.KAWAY_Z+1)
        self.NUM_NEIGHBORS = (self.NBRS_X * self.NBRS_Y * self.NBRS_Z)

        self.CELLARRAY_DIM_X = 3
        self.CELLARRAY_DIM_Y = 3
        self.CELLARRAY_DIM_Z = 3
        self.PTP_CUT_OFF = 26 # cut off for atom to atom interactions
        self.CELL_MARGIN = 4  # constant diff between cutoff and cell size
        self.CELL_SIZE_X = (self.PTP_CUT_OFF + self.CELL_MARGIN)//self.KAWAY_X
        self.CELL_SIZE_Y = (self.PTP_CUT_OFF + self.CELL_MARGIN)//self.KAWAY_Y
        self.CELL_SIZE_Z = (self.PTP_CUT_OFF + self.CELL_MARGIN)//self.KAWAY_Z

        self.cellArrayDimX = self.CELLARRAY_DIM_X
        self.cellArrayDimY = self.CELLARRAY_DIM_Y
        self.cellArrayDimZ = self.CELLARRAY_DIM_Z


        # variables to control initial uniform placement of atoms;
        # atoms should not be too close at startup for a stable system
        # PERDIM * GAP should be less than (PTPCUTOFF+CELL_MARGIN)
        # max particles per cell should not be greater thatn PERDIM^3 for 1 AWAY
        self.PERDIM = 10
        self.GAP = 3

        self.CELL_ORIGIN_X = 0
        self.CELL_ORIGIN_Y = 0
        self.CELL_ORIGIN_Z = 0

        self.MIGRATE_STEPCOUNT = 20
        self.DEFAULT_FINALSTEPCOUNT = 1001
        self.MAX_VELOCITY = .1  # in A/fs

        self.finalStepCount = self.DEFAULT_FINALSTEPCOUNT
        self.firstLdbStep = self.DEFAULT_FIRST_LDB
        self.ldbPeriod = self.DEFAULT_LDB_PERIOD


        # Proxies for the different arrays
        self.cellArray = None
        self.computeArray = None

def WRAP_X(a):
    return (a + cellArrayDimX) % cellArrayDimX
def WRAP_Y(a):
    return (a + cellArrayDimY) % cellArrayDimY
def WRAP_Z(a):
    return (a + cellArrayDimZ) % cellArrayDimZ


@numba.njit( cache = True )
def velocityCheck( inVelocity: float ) -> float:
    if abs( inVelocity ) > MAX_VELOCITY:
        if inVelocity < 0.0:
            return -1 * MAX_VELOCITY
        return MAX_VELOCITY
    return inVelocity


@numba.njit( cache = True )
def updateProperties( forces, particle_mass, particle_vel, particle_pos,
                      energy, stepCount, finalStepCount ):
    powTen = 10.0 ** 10
    powTwenty = 10.0 ** -20
    realTimeDeltaVel = DEFAULT_DELTA * powTwenty
    for i in range(particle_mass.size):
        mass = particle_mass[i]
        # calculate energy only at beginning and end
        if (stepCount == 1):
            dot = particle_vel[i,0]**2 + particle_vel[i,1]**2 + particle_vel[i,2]**2
            energy[0] += (0.5 * mass * dot * powTen)  # in milliJoules
        elif (stepCount == finalStepCount):
            dot = particle_vel[i,0]**2 + particle_vel[i,1]**2 + particle_vel[i,2]**2
            energy[1] += (0.5 * mass * dot * powTen)
            # apply kinetic equations
        invMassParticle = 1.0 / mass
            #self.particles[i].acc = forces[i] * invMassParticle  # in m/sec^2
            #self.particles[i].vel += self.particles[i].acc * realTimeDeltaVel  # in A/fs
            # in m/sec^2
        particle_vel[i,0] += forces[i,0] * invMassParticle * realTimeDeltaVel  # in A/fs
        particle_vel[i,1] += forces[i,1] * invMassParticle * realTimeDeltaVel  # in A/fs
        particle_vel[i,2] += forces[i,2] * invMassParticle * realTimeDeltaVel  # in A/fs

        particle_vel[i,0] = velocityCheck(particle_vel[i,0])
        particle_vel[i,1] = velocityCheck(particle_vel[i,1])
        particle_vel[i,2] = velocityCheck(particle_vel[i,2])

        particle_pos[i,0] += particle_vel[i,0] * DEFAULT_DELTA  # in A
        particle_pos[i,1] += particle_vel[i,1] * DEFAULT_DELTA  # in A
        particle_pos[i,2] += particle_vel[i,2] * DEFAULT_DELTA  # in A

class CellMap( ArrayMap ):
    # group
    def __init__( self, cellX, cellY, cellZ ):
        self.num_x = cellX
        self.num_y = cellY
        self.num_z = cellZ

        self.num_yz = self.num_y * self.num_z
        self.ratio = charm.numPes() / ( self.num_x * self.num_yz )

    def procNum( self, index ):
        patchID = index[ 2 ] + index[ 1 ] * self.num_z + index[ 0 ] * self.num_yz
        return int( patchID * self.ratio )

class Particle:
    def __init__( self ):
        self.mass = 0.0
        self.position = np.zeros( 3 )
        self.acceleration = np.zeros( 3 )
        self.velocity = np.zeros( 3 )

class Cell( Chare ):

    def __init__( self, energyFuture ):
        self.stepCount :int = 0
        self.mynumParts :int = 0
        self.inbrs :int = NUM_NEIGHBORS
        self.stepTime: float = 0
        self.computesList = [0] * self.inbrs
        self.neighborChannels = list()
        self.updateCount :int = 0
        self.duplicateComputes = None
        self.energy = np.zeros(2, dtype = np.float64)
        self.mCastSecProxy = None

        self.energyFuture = energyFuture
        self.myid: int = self.thisIndex[ 2 ] + cellArrayDimZ * \
                    ( self.thisIndex[1] + self.thisIndex[0] * cellArrayDimY)

        num = self.myid * (PARTICLES_PER_CELL_END-PARTICLES_PER_CELL_START)
        denom = cellArrayDimX*cellArrayDimY*cellArrayDimZ
        self.myNumParts = PARTICLES_PER_CELL_START + ( num // denom )

        self.particle_mass = np.zeros( self.myNumParts, dtype = np.float64 )
        self.particle_pos = np.zeros( ( self.myNumParts, 3 ), dtype = np.float64 )
        self.particle_vel = np.zeros( ( self.myNumParts, 3 ), dtype = np.float64 )

        self.neighborChannels = self.createNeighborChannels()

        random.seed( self.myid )

        for i in range( self.myNumParts ):
            self.particle_mass[ i ] = HYDROGEN_MASS

            # uniformly place particles, avoid close distance among them
            x = (GAP/2.0) + self.thisIndex[0] * CELL_SIZE_X + ((i*KAWAY_Y*KAWAY_Z)//(PERDIM*PERDIM))*GAP
            y = (GAP/2.0) + self.thisIndex[1] * CELL_SIZE_Y + (((i*KAWAY_Z)//PERDIM)%(PERDIM//KAWAY_Y))*GAP
            z = (GAP/2.0) + self.thisIndex[2] * CELL_SIZE_Z + (i%(PERDIM//KAWAY_Z))*GAP
            self.particle_pos[ i ] = x, y, z

            self.particle_vel[i] = np.array( ( (random.random() - 0.5) * .2 * MAX_VELOCITY,
                                               (random.random() - 0.5) * .2 * MAX_VELOCITY,
                                               (random.random() - 0.5) * .2 * MAX_VELOCITY),
                                            dtype = np.float64
            )

        self.energy[ 0 ] = 0
        self.energy[ 1 ] = 0

    def reportDuplicates( self ):
        for d in self.duplicateComputes:
            computeArray[ d ].setDuplicate()

    def nbrNumtoNbrIdx( self, num ):
        x1 = num // (NBRS_Y * NBRS_Z) - NBRS_X // 2
        y1 = (num % (NBRS_Y * NBRS_Z)) // NBRS_Z - NBRS_Y // 2
        z1 = num % NBRS_Z - NBRS_Z // 2

        return ( WRAP_X( self.thisIndex[ 0 ] + x1 ),
                 WRAP_Y( self.thisIndex[ 1 ] + y1 ),
                 WRAP_Z( self.thisIndex[ 2 ] + z1 )
        )

    @coro
    def createNeighborChannels( self ):
        output = list()
        for num in range( self.inbrs ):
            nbrIdx = self.nbrNumtoNbrIdx( num )
            output.append( Channel( self, remote = self.thisProxy[ nbrIdx ] ) )
        return output

    def createComputes( self ):
        x, y, z = self.thisIndex

        currPe = charm.myPe() + 1

        dupes = list()
        seen = set()

        for num in range( self.inbrs ):
            dx = num // ( NBRS_Y * NBRS_Z ) - NBRS_X // 2
            dy = ( num % ( NBRS_Y * NBRS_Z ) ) // NBRS_Z - NBRS_Y // 2
            dz = num % NBRS_Z - NBRS_Z // 2

            if num >= self.inbrs // 2:
                px1 = x + KAWAY_X
                py1 = y + KAWAY_Y
                pz1 = z + KAWAY_Z

                px2 = px1 + dx
                py2 = py1 + dy
                pz2 = pz1 + dz

                currPe += 1

                # CkArrayIndex6D index(px1, py1, pz1, px2, py2, pz2);
                index = ( px1, py1, pz1, px2, py2, pz2 )
                computeArray.ckInsert( index, onPE = ( currPe ) % charm.numPes(),
                                       args = [ self.energyFuture ], useAtSync = True
                )
                self.computesList[num] = index
            else:
                px1 = WRAP_X(x + dx) + KAWAY_X
                py1 = WRAP_Y(y + dy) + KAWAY_Y
                pz1 = WRAP_Z(z + dz) + KAWAY_Z
                px2 = px1 - dx
                py2 = py1 - dy
                pz2 = pz1 - dz
                index = (px1, py1, pz1, px2, py2, pz2)
                self.computesList[num] = index

        for c in self.computesList:
            if c in seen:
                dupes.append( c )
            seen.add(c)

        self.computesList = list(seen)
        self.duplicateComputes = dupes

    def migrateToCell( self, particlePos ):
        x = self.thisIndex[ 0 ] * CELL_SIZE_X + CELL_ORIGIN_X
        y = self.thisIndex[ 1 ] * CELL_SIZE_Y + CELL_ORIGIN_Y
        z = self.thisIndex[ 2 ] * CELL_SIZE_Z + CELL_ORIGIN_Z

        px = py = pz = 0
        particleXpos = particlePos[ 0 ]
        particleYpos = particlePos[ 1 ]
        particleZpos = particlePos[ 2 ]

        if particleXpos < (x-CELL_SIZE_X):
            px = -2
        elif particleXpos < x:
            px = -1
        elif particleXpos > (x+2*CELL_SIZE_X):
            px = 2
        elif particleXpos > (x+CELL_SIZE_X):
            px = 1

        if particleYpos < (y-CELL_SIZE_Y):
            py = -2
        elif particleYpos < y:
            py = -1
        elif particleYpos > (y+2*CELL_SIZE_Y):
            py = 2
        elif particleYpos > (y+CELL_SIZE_Y):
            py = 1

        if particleZpos < (z-CELL_SIZE_Z):
            pz = -2
        elif particleZpos < z:
            pz = -1
        elif particleZpos > (z+2*CELL_SIZE_Z):
            pz = 2
        elif particleZpos > (z+CELL_SIZE_Z):
            pz = 1

        return ( px, py, pz ) # setting px, py, pz to zero

    def wrapAround( self, particlePos ):
        if particlePos[ 0 ] < CELL_ORIGIN_X:
            particlePos[ 0 ] += CELL_SIZE_X*cellArrayDimX
        if particlePos[ 1 ] < CELL_ORIGIN_Y:
            particlePos[ 1 ] += CELL_SIZE_Y*cellArrayDimY
        if particlePos[ 2 ] < CELL_ORIGIN_Z:
            particlePos[ 2 ] += CELL_SIZE_Z*cellArrayDimZ

        if particlePos[ 0 ] > CELL_ORIGIN_X + CELL_SIZE_X*cellArrayDimX:
            particlePos[ 0 ] -= CELL_SIZE_X*cellArrayDimX
        if particlePos[ 1 ] > CELL_ORIGIN_Y + CELL_SIZE_Y*cellArrayDimY:
            particlePos[ 1 ] -= CELL_SIZE_Y*cellArrayDimY
        if particlePos[ 2 ] > CELL_ORIGIN_Z + CELL_SIZE_Z*cellArrayDimZ:
            particlePos[ 2 ] -= CELL_SIZE_Z*cellArrayDimZ
        return particlePos

    def createSection( self ):
        # computeArray is global
        self.mCastSecProxy = charm.split( computeArray, 1, elems = [ self.computesList ] )[ 0 ]

    @coro
    def migrateParticles( self ):
        outgoing = [ [[],[],[]] for _ in range(self.inbrs) ]

        size = numParts = self.particle_mass.size

        for i in range(numParts - 1, -1 -1 ):
            x1, y1, z1 = self.migrateToCell( self.particle_pos[ i ] )
            if any( [x1, y1, z1 ] ):
                outIndex = (x1+KAWAY_X)*NBRS_Y*NBRS_Z + (y1+KAWAY_Y)*NBRS_Z + (z1+KAWAY_Z)

                outgoing[outIndex][0].append(self.particle_mass[i])
                outgoing[outIndex][1].append(self.wrapAround(self.particle_pos[i].copy()))
                outgoing[outIndex][2].append(self.particle_vel[i].copy())
                self.particle_mass[i] = self.particle_mass[size-1]
                self.particle_pos[i]  = self.particle_pos[size-1]
                self.particle_vel[i]  = self.particle_vel[size-1]
                size -= 1


        if size < numParts:
            self.particle_mass = self.particle_mass[:size].copy()
            self.particle_pos  = self.particle_pos[:size].copy()
            self.particle_vel  = self.particle_vel[:size].copy()


        for num in range( self.inbrs ):
            numOutgoing = len(outgoing[num][0])
            if numOutgoing > 0:
                mass = np.array(outgoing[num][0], dtype=np.float64)
                pos  = np.concatenate(outgoing[num][1])
                vel  = np.concatenate(outgoing[num][2])
                self.neighborChannels[ num ].send(True, mass, pos, vel)
            else:
                self.neighborChannels[ num ].send(True, None, None, None)


    def sendPositions( self, forceFuture ):
        self.mCastSecProxy.calculateForces( self.mCastSecProxy,
                                            np.array(self.thisIndex),
                                            self.particle_pos, forceFuture
        )

    def resumeFromSync(self):
        if not any( self.thisIndex ):
            stepT = time.time()
            print( f'Step {self.stepCount} Time {(stepT-self.stepTime)*1000} ms/step' )
            self.stepTime = stepT
        self.thisProxy[ self.thisIndex ].run()


    @coro
    def run( self ):
        if self.stepCount == 0:
            self.reportDuplicates()
            self.createSection()
            self.stepCount = 1

        # todo: something not quite right here
        if not any( self.thisIndex ):
            self.stepTime = time.time()


        for self.stepCount in range( self.stepCount, finalStepCount + 1 ):
            reduceForceFuture = Future()
            self.sendPositions( reduceForceFuture )
            forces = reduceForceFuture.get()
            updateProperties( forces, self.particle_mass, self.particle_vel,
                              self.particle_pos, self.energy, self.stepCount,
                              finalStepCount
            )

            if not self.stepCount % MIGRATE_STEPCOUNT:
                self.migrateParticles()
                for ch in charm.iwait( self.neighborChannels ):
                    self.receiveParticles( *ch.recv() )

            # TODO: Add a check to see if load balancing should be done here
            if self.shouldLoadBalance(): 
                self.AtSync()
                return

            if not any( self.thisIndex ):
                stepT = time.time()
                print( f'Step {self.stepCount} Time {(stepT-self.stepTime)*1000} ms/step' )
                self.stepTime = stepT
        self.reduce( self.energyFuture, self.energy, Reducer.sum )

    def shouldLoadBalance( self ):
        return not any( [ self.stepCount <= firstLdbStep, self.stepCount % ldbPeriod, self.stepCount >= finalStepCount ] )

    def receiveParticles( self, r, mass, poss, vel ):
        if mass is not None:
            total = self.particle_mass.size + mass.size
            self.particle_mass = np.append(self.particle_mass, mass)
            self.particle_pos  = np.append(self.particle_pos, pos)
            self.particle_vel  = np.append(self.particle_vel, vel)
            self.particle_pos.shape = (total, 3)
            self.particle_vel.shape = (total, 3)


class Physics:

    @numba.njit( cache = True )
    def calcPairForces( firstIndex, secondIndex,
                        firstPos, secondPos,
                        stepCount,
                        force1, force2
    ) -> float:

        firstLen = firstPos.shape[0]
        secondLen = secondPos.shape[0]
        energy = 0.0
        doEnergy = False
        if stepCount == 1 or stepCount == finalStepCount:
            doEnergy = True

        # check for wrap around and adjust locations accordingly
        diff_0, diff_1, diff_2 = 0.0, 0.0, 0.0
        if abs(firstIndex[0] - secondIndex[0]) > 1 :
            diff_0 = CELL_SIZE_X * cellArrayDimX
            if secondIndex[0] < firstIndex[0] : diff_0 = -1 * diff_0
        if abs(firstIndex[1] - secondIndex[1]) > 1 :
            diff_1 = CELL_SIZE_Y * cellArrayDimY
            if secondIndex[1] < firstIndex[1] : diff_1 = -1 * diff_1

        if abs(firstIndex[2] - secondIndex[2]) > 1 :
            diff_2 = CELL_SIZE_Z * cellArrayDimZ
            if secondIndex[2] < firstIndex[2] : diff_2 = -1 * diff_2

        ptpCutOffSqd = PTP_CUT_OFF * PTP_CUT_OFF
        powTen = 10.0 ** -10
        powTwenty = 10.0 ** -20

        separation_0, separation_1, separation_2 = 0.0, 0.0, 0.0
        for i1 in range(0, firstLen, BLOCK_SIZE):
            for j1 in range(0, secondLen, BLOCK_SIZE):
                for i in range(i1, min(i1+BLOCK_SIZE, firstLen)):
                    for j in range(j1, min(j1+BLOCK_SIZE, secondLen)):
                        #separation = firstPos[i] - secondPos[j]
                        separation_0 = firstPos[i,0] + diff_0 - secondPos[j,0]
                        separation_1 = firstPos[i,1] + diff_1 - secondPos[j,1]
                        separation_2 = firstPos[i,2] + diff_2 - secondPos[j,2]
                        rsqd = separation_0**2 + separation_1**2 + separation_2**2
                        #rsqd = dot(separation, separation)
                        if rsqd > 1 and rsqd < ptpCutOffSqd:
                            rsqd = rsqd * powTwenty
                            r = math.sqrt(rsqd)
                            rSix = rsqd * rsqd * rsqd
                            rTwelve = rSix * rSix
                            f = ( (12 * VDW_A) / rTwelve - (6 * VDW_B) / rSix)
                            if doEnergy:
                                energy += ( VDW_A / rTwelve - VDW_B / rSix)  # in milliJoules
                                fr = f / rsqd
                                #force = separation * (fr * powTen)
                                #force1[i] += force
                                #force2[j] -= force
                                force_0 = separation_0 * (fr * powTen)
                                force_1 = separation_1 * (fr * powTen)
                                force_2 = separation_2 * (fr * powTen)
                                force1[i,0] += force_0
                                force1[i,1] += force_1
                                force1[i,2] += force_2
                                force2[j,0] -= force_0
                                force2[j,1] -= force_1
                                force2[j,2] -= force_2

        return energy

    @numba.njit( cache = True )
    def calcInternalForces( firstPos, firstIndex, stepCount, force1 ):
        firstLen = firstPos.shape[0]
        energy = 0.0
        doEnergy = False
        if (stepCount == 1 or stepCount == finalStepCount):
            doEnergy = True

        ptpCutOffSqd = PTP_CUT_OFF * PTP_CUT_OFF
        powTen = 10.0 ** -10
        powTwenty = 10.0 ** -20
        separation_0, separation_1, separation_2 = 0.0, 0.0, 0.0
        force_0, force_1, force_2 = 0.0, 0.0, 0.0
        for i in range(firstLen) :
            for j in range(i+1, firstLen) :
                # computing base values
                separation_0 = firstPos[i,0] - firstPos[j,0]
                separation_1 = firstPos[i,1] - firstPos[j,1]
                separation_2 = firstPos[i,2] - firstPos[j,2]
                rsqd = separation_0**2 + separation_1**2 + separation_2**2
                if rsqd > 1 and rsqd < ptpCutOffSqd:
                    rsqd = rsqd * powTwenty
                    r = math.sqrt(rsqd)
                    rSix = rsqd * rsqd * rsqd
                    rTwelve = rSix * rSix
                    f = ( (12 * VDW_A) / rTwelve - (6 * VDW_B) / rSix)
                    if(doEnergy) :
                        energy += ( VDW_A / rTwelve - VDW_B / rSix)

                    fr = f / rsqd
                    force_0 = separation_0 * (fr * powTen)
                    force_1 = separation_1 * (fr * powTen)
                    force_2 = separation_2 * (fr * powTen)
                    force1[i,0] += force_0
                    force1[i,1] += force_1
                    force1[i,2] += force_2
                    force1[j,0] -= force_0
                    force1[j,1] -= force_1
                    force1[j,2] -= force_2

        return energy



class Compute( Chare ):
    def __init__( self, energySumFuture = None ):
        self.energy = np.zeros(2, dtype=np.float64)
        self.stepCount = 1
        self.energySumFuture = energySumFuture

        self.dataReceived = list()

        self.isDuplicate = False
        self._self_compute = None


    def isSelfCompute( self ):
        if self._self_compute is None:
            conds = [ self.thisIndex[ x ] == self.thisIndex[ x + 3 ] for x in range( len( self.thisIndex ) // 2 ) ]
            self._self_compute = all( conds )
        return self._self_compute


    def setDuplicate(self):
        self.isDuplicate = True

    def calculateForces( self, secProxy, senderCoords, forces, doneFut ):
        if self.isSelfCompute():
            self.selfInteract( secProxy, senderCoords, forces, doneFut )
            self.stepCount += 1
        else:
            self.dataReceived.append( [ secProxy, senderCoords, forces, doneFut ] )
            assert len( self.dataReceived ) < 3

            if self.isDuplicate:
                # Not all neighbors are unique, we treat the duplicates as
                # self interactions, but we have to receive both duplicates.
                self.selfInteract( secProxy, senderCoords, forces, doneFut )
                self.dataReceived = list()
            elif len( self.dataReceived ) == 2:
                redProxy1, coords1, forces1, doneFut1 = self.dataReceived[ 0 ]
                redProxy2, coords2, forces2, doneFut2 = self.dataReceived[ 1 ]
                self.thisProxy[self.thisIndex].interact( redProxy1, coords1, forces1, doneFut1, redProxy2, coords2, forces2, doneFut2 )
                self.dataReceived = list()
            self.stepCount += 1

        if self.stepCount > finalStepCount:
            # Everything done, reduction on potential energy
            assert len( self.energy ) == 2
            self.reduce( self.energySumFuture, self.energy, Reducer.sum )

        # TODO: Add a check to see if load balancing should be done here
        if self.stepCount > firstLdbStep and not self.stepCount % ldbPeriod:
            self.AtSync()

    def resumeFromSync(self):
        # this is a reactive chare, so it doesn't do anything on resume.
        # Still, this method must exist in the chare
        pass

    def selfInteract( self, mcast1, senderCoords, msg, doneFuture ):
        energyP: float = 0

        force1 = np.zeros( (len(msg),3), dtype = np.float64 )

        energyP = Physics.calcInternalForces( msg, senderCoords, self.stepCount, force1 )

        if self.stepCount == 1:
            self.energy[ 0 ] = energyP
        elif self.stepCount == finalStepCount:
            self.energy[ 1 ] = energyP

        self.contribute( force1, Reducer.sum, doneFuture, mcast1 )

    def setReductionClient( self, proxy, method ):
        self.reductionClientProxy = proxy
        self.reductionClientMethod = method
        self.reductionClientFn = getattr( proxy, method )

    def interact( self, mcast1, coords1, msg1, doneFut1,
                  mcast2, coords2, msg2, doneFut2
    ):
        x1, y1, z1 = coords1
        x2, y2, z2 = coords1
        doSwap = False
        if x2 * cellArrayDimY * cellArrayDimZ + y2 * cellArrayDimZ + z2 < \
           x1 * cellArrayDimY * cellArrayDimZ + y1 * cellArrayDimZ + z1:
            mcast1, mcast2 = mcast2, mcast1
            doneFut1, doneFut2 = doneFut2, doneFut1
            doSwap = True

        # unpacking arguments so they can be sent to the numba calcPairForces
        force1 = np.zeros( ( len(msg1), 3 ), dtype = np.float64 )
        force2 = np.zeros( ( len(msg2), 3 ), dtype = np.float64 )
        energyP = Physics.calcPairForces( coords1, coords2,
                                          msg1,
                                          msg2,
                                          self.stepCount,
                                          force1,
                                          force2
        )

        if doSwap:
            force1, force2 = force2, force1

        if self.stepCount == 1:
            self.energy[ 0 ] = energyP
        elif self.stepCount == finalStepCount:
            self.energy[ 1 ] = energyP

        self.reduce( doneFut1, force1, Reducer.sum, mcast1 )
        self.reduce( doneFut2, force2, Reducer.sum, mcast2 )


def energySum( startEnergy, endEnergy ):
    iE1, fE1 = startEnergy
    iE2, fE2 = endEnergy
    if abs( fE1 + fE2 - iE1 - iE2 ) > ENERGY_VAR:
        print( f'Energy value has varied significantly from {iE1+iE2} to {fE1 + fE2}' )
    else:
        print( 'Energy conservation test passed for maximum allowed variation of '
               f'{ENERGY_VAR} units. \nSIMULATION SUCCESSFUL'
        )


def main( args ):
    print( 'LENNARD JONES MOLECULAR DYNAMICS START UP...' )
    Chare( Compute )

    if len( args ) != 7:
        print( 'USAGE python3 -m charmrun.start +p<NProcs> dimX dimY dimZ steps firstLBstep LBPeriod' )
        exit()

    globs = GlobalDefs()

    dimX, dimY, dimZ = [ int( x ) for x in args[ 1:4 ] ]
    globs.cellArrayDimX, globs.cellArrayDimY, globs.cellArrayDimZ = dimX, dimY, dimZ
    steps = int( args[ 4 ] )
    globs.finalStepCount = steps
    globs.firstLdbStep = int( args[ 5 ] )
    globs.lbPeriod = int( args[ 6 ] )

    print( f'Cell Array Dimension X: {dimX} Y: {dimY} Z: {dimZ} '
           f'of size {globs.CELL_SIZE_X} {globs.CELL_SIZE_Y} {globs.CELL_SIZE_Z}'
    )
    print( f'Final Step Count: {steps}' )
    print( f'First LB Step: {globs.firstLdbStep}' )
    print( f'LB Period: {globs.lbPeriod}' )

    charm.thisProxy.updateGlobals( globs.__dict__, awaitable = True ).get()

    doneFuture = Future()

    # 2, one for start energy and one for end energy
    energyFuture = Future( 2 )

    cellMap = Group( CellMap, args = ( dimX, dimY, dimZ ) )
    globs.cellArray = Array( Cell, ( dimX, dimY, dimZ ), map = cellMap, args = [ energyFuture ], useAtSync = True )
    globs.computeArray = Array( Compute, ndims = 6 )
    charm.thisProxy.updateGlobals( globs.__dict__, awaitable = True ).get()
    globs.cellArray.createComputes( awaitable = True ).get()
    charm.thisProxy.updateGlobals( globs.__dict__, awaitable = True ).get()

    print( f'Cells: {globs.cellArrayDimY} X {globs.cellArrayDimY} X {globs.cellArrayDimZ} .... created' )

    computeArray.ckDoneInserting()

    nComputes = (NUM_NEIGHBORS//2+1) * \
                cellArrayDimX*cellArrayDimY*cellArrayDimZ
    print(f"Computes: {nComputes} .... created\n" )
    print("Starting simulation .... \n\n")

    startBenchmarkTime = time.time()


    cellArray.run()
    starting, ending = energyFuture.get()

    energySum( starting, ending )

    endBenchmarkTime = time.time()

    print( f'Total application time: {endBenchmarkTime - startBenchmarkTime}' )
    exit()


if __name__ == '__main__':
    charm.start( main )
