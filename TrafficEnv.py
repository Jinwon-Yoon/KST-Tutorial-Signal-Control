
import numpy as np
import traci



class TrafficEnv:
    
    
    def __init__(self):
        
        self.sumoBinary = "D:/SUMO/bin/sumo-gui"        ############# 경로는 개인별로 설정!!!!
        self.cfgBinary = "tutorial.sumocfg"             ############# 경로는 개인별로 설정!!!!
        self.sumoCmd = [self.sumoBinary, "-c", self.cfgBinary]
        
        self.timestep = -1
        self.period = 240
        self.edgeId = ["3to1", "5to1", "4to1", "2to1"]
        
        self.outEdgeId = ["1to3", "1to5", "1to4", "1to2"]
        self.outQIds = [[], [], [], []]
        self.QIds = [[], [], [], []]
        self.flow = 300
        self.waitingTimes = [[], [], [], []]
        
        self.stateDim = 4
        self.periodicState = np.zeros((self.period, self.stateDim))
        self.state = None
        
        ## Traffic Signal
        self.cycle = 120
        self.signalTime = [[], []]
        
        
    def startSUMO(self):
        traci.start(self.sumoCmd)
        
    def endSUMO(self):
        traci.close()
        
    def stepSimulation(self):
        traci.simulationStep()
        self.timestep += 1
        
        
    def reset(self):
        if self.timestep == -1:
            prdState = self.periodicState
            for t in range(self.period):
                self.stepSimulation()
                prdState[t, :] = self.getInstantQs()
            
            self.state = self.getMaxQs(prdState)
        else:
            print("Error! You may close the current Simulation.")
        return self.state
        
            
    def getInstantQs(self):
        edgeQs = []
        for edge in self.edgeId:
            edgeVehIds = traci.edge.getLastStepVehicleIDs(edge)
            edgeQ = 0
            if len(edgeVehIds) != 0:
                for idv in edgeVehIds:
                    vehPos = traci.vehicle.getLanePosition(idv)
                    vehSpd = traci.vehicle.getSpeed(idv)
                    if vehPos > 100 and vehSpd < 10:
                        edgeQ += 1
            edgeQs.append(edgeQ)
        return edgeQs


    def getMaxQs(self, periodicState):
        maxQs = []
        for i in range(self.stateDim):
            maxQ = np.max(periodicState[range(self.cycle, len(periodicState)), i])
            maxQs.append(maxQ)
        return maxQs
        
    
    def setAction(self, action):
        '''
        action is the green time ratio of N-S direction (0.2, 0.4, 0.6, 0.8)
        cycle is fixed to be 120 (sec)
        and offset is set to be zero
        '''
        gr = (action + 1) * 0.2
        
        greentime_NS = int(round(self.cycle * gr * 1000))
        greentime_EW = int(self.cycle * 1000 - greentime_NS)
        
        phase = []
        phase.append(traci.trafficlights.Phase(greentime_EW, 0, 0, "rrrGGGrrrGGG"))
        phase.append(traci.trafficlights.Phase(greentime_NS, 0, 0, "GGGrrrGGGrrr"))
        
        programId = ''.join(['action-', str(gr)])
        logic = traci.trafficlights.Logic(programId, 0, 0, 0, phase)
        traci.trafficlights.setCompleteRedYellowGreenDefinition('1', logic)
        
        
    def extractSignalTimes(self):
        
        currentPhase = traci.trafficlight.getPhase('1')
        currentPhaseDur = traci.trafficlight.getPhaseDuration('1')
        
        prevPhaseDur = self.signalTime[currentPhase]
        
        if prevPhaseDur != currentPhaseDur:
            self.signalTime[currentPhase] = currentPhaseDur



    def getInstantThroughput(self):
        '''
        Get instant throughput by counting the number of vehicles in outflow links
        in a one-simulation tic
        '''
        for i in range(0, 4):
            edge = self.outEdgeId[i]
            edgeVehIds = traci.edge.getLastStepVehicleIDs(edge)
            self.outQIds[i].append(edgeVehIds)
            
        return self.outQIds
    
    
    def getReward(self):
        
        totalThroughput = []
        for i in range(0, 4):
            ids = sum(self.outQIds[i], [])
            dirThroughput = len(list(set(ids)))
            totalThroughput.append(int(dirThroughput))
        
        reward = sum(totalThroughput) / self.flow
                    
        self.outQIds = [[], [], [], []]
        
        return reward
            
            

    
    def getTerminal(self):
        terminal = False
        return terminal


    def step(self, action):
                
        # Set signal action
        self.setAction(action)
        
        # Proceed 1 simulation step (120 sec)
        prdState = self.periodicState
        for t in range(self.period):
            self.stepSimulation()
            # Get next state
            prdState[t, :] = self.getInstantQs()
            
            # Get reward
            self.getInstantThroughput()
            #self.new_getInstantReward()
            
            # Extract Signal Time Info
            self.extractSignalTimes()
            
        # Get next State
        self.state_raw = prdState
        self.state = self.getMaxQs(prdState)
        nextState = self.state
        
        # Get reward
        reward = self.getReward()
        #reward = self.new_getReward()
        
        # Get termination
        terminal = self.getTerminal()
        
        return nextState, reward, terminal
    
    
    
    
    
    def new_getInstantReward(self):
        
        for i in range(0, 4):
            edge = self.edgeId[i]
            edgeWaitingTimes = traci.edge.getWaitingTime(edge)
            edgeVehicleNumbers = traci.edge.getLastStepVehicleNumber(edge)
            self.waitingTimes[i].append(edgeWaitingTimes / edgeVehicleNumbers)  # Get average waiting times
            
        return self.waitingTimes
        
        
    def new_getReward(self):
        
        totalWaitingTimes = 0
        for i in range(0, 4):
            totalWaitingTimes += sum(self.waitingTimes[i])
        
        reward = round(-(totalWaitingTimes - 5000) / 10000, 2)
        
        if reward > 1:
            reward = 1
        elif reward < -1:
            reward = -1
            
        self.waitingTimes = [[], [], [], []]
            
        return reward
        
        