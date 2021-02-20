

def add(name, function, model, inputs: str, location: str):

    model_name = ""
    if inputs == "text":
        from unbox.templates.tensorflow.text.template import TensorflowTextTemplateModel as TemplateModel
        model_name = "TensorflowTextTemplateModel"
    elif inputs == "image":
        from unbox.templates.tensorflow.image.template import TensorflowImageTemplateModel as TemplateModel
        model_name = "TensorflowImageTemplateModel"

    bento_service = TemplateModel()
    bento_service.pack('model', model)
    bento_service.pack('function', function)
    saved_path = bento_service.save()

    return saved_path.split("/")[-1], model_name
