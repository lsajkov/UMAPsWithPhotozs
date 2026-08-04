from rail.core.stage import RailStage
from rail.core.data  import PqHandle
from ceci.config     import StageParameter
from photerr         import ErrorModel, ErrorParams

class MultiSurveyErrorModel(RailStage):
    
    name = "MultiSurveyErrorModel"
    
    inputs  = [("noiseless_catalog", PqHandle)]
    outputs = [("noisy_catalog",     PqHandle)]

    config_options = RailStage.config_options.copy()

    config_options.update(dict(
        inputType  = StageParameter(str, 'pogson'),
        outputType = StageParameter(str, 'asinh'),
        m5         = StageParameter(dict, {}),
        nYrObs     = StageParameter(float, 1.0),
        nVisYr     = StageParameter(float, 1.0),
        gamma      = StageParameter(float, 1.0),
        sigLim     = StageParameter(float, 1.0),
        seed       = StageParameter(int, 42)
    ))
    
    def run(self):
    
        noiseless_catalog = self.get_data("noiseless_catalog")
        
        params = ErrorParams(inputType  = self.config.inputType,
                             outputType = self.config.outputType,
                             m5         = self.config.m5,
                             nYrObs     = self.config.nYrObs,
                             nVisYr     = self.config.nVisYr,
                             gamma      = self.config.gamma,
                             sigLim     = self.config.sigLim)
        errorModel = ErrorModel(params)
        
        noisy_catalog = errorModel(noiseless_catalog, random_state = self.config.seed)
        
        self.add_data("noisy_catalog", noisy_catalog)