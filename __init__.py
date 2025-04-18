from .TSPPostProcessor import TSPPostProcessor

def getMetaData():
    from cura.CuraApplication import CuraApplication
    return {
        "mesh": CuraApplication.getInstance().getMetaData().get("mesh")
    }

def register(app):
    return { "postprocessor": TSPPostProcessor(app) }
