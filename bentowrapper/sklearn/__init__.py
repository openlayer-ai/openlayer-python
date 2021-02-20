

def add(name, function, model, inputs: str, location: str):

    model_name = ""
    if inputs == "text":
        from bentowrapper.templates.sklearn.text.template import SklearnTextTemplateModel as TemplateModel
        model_name = "SklearnTextTemplateModel"
    elif inputs == "image":
        from bentowrapper.templates.sklearn.image.template import SklearnImageTemplateModel as TemplateModel
        model_name = "SklearnImageTemplateModel"

    bento_service = TemplateModel()
    bento_service.pack('model', model)
    bento_service.pack('function', function)
    saved_path = bento_service.save()

    return saved_path.split("/")[-1], model_name
