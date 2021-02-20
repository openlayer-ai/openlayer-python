

def add(name, function, model, inputs: str, location: str):

    model_name = ""
    if inputs == "text":
        from unbox.templates.pytorch.text.template import PytorchTextTemplateModel as TemplateModel
        model_name = "PytorchTextTemplateModel"
    elif inputs == "image":
        from unbox.templates.pytorch.image.template import PytorchImageTemplateModel as TemplateModel
        model_name = "PytorchImageTemplateModel"

    bento_service = TemplateModel()
    bento_service.pack('model', model)
    bento_service.pack('function', function)
    saved_path = bento_service.save()

    return saved_path.split("/")[-1], model_name
