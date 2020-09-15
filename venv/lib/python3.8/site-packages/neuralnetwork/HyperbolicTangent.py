import math

class HyperbolicTangent:
    def getActivation(self,net):
        return math.tanh(net)

    def getDerivative(self,net):
        return 1 - (self.getActivation(net) * self.getActivation(net))