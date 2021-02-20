

def add(name, function, model, inputs: str, location: str):
    if inputs == "text":
        from bentowrapper.templates.transformers.text.template import TransformersTextTemplateModel as TemplateModel

    bento_service = TemplateModel()
    bento_service.pack('model', model)
    bento_service.pack('function', function)
    saved_path = bento_service.save()

    return saved_path.split("/")[-1], "TransformersTextTemplateModel"