only_model_string = '''
        @artifacts([{0}('model'), PickleArtifact('function'), PickleArtifact('active_learning_function')])
        class TemplateModel(BentoService):

            @api(input=JsonInput(), batch=True)
            def predict(self, parsed_json_list: List[JsonSerializable]):
                text = []
                for json in parsed_json_list:
                    if 'text' in json:
                        text.append(json['text'])
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')

                prediction_probs, class_names, _ = self.artifacts.function(
                    self.artifacts.model,
                    text
                )

                return [
                    {{class_names[i]: prob for i,
                        prob in enumerate(probs)}}
                    for probs in prediction_probs
                ]

            @api(input=JsonInput(), batch=True)
            def predictbatch(self, parsed_json_list: List[JsonSerializable]):
                text = []
                for json in parsed_json_list:
                    if 'batch' in json:
                        text = json['batch']
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')
                prediction_probs, class_names, _ = self.artifacts.function(
                    self.artifacts.model,
                    text
                )

                return [[
                    {{class_names[i]: prob for i,
                        prob in enumerate(probs)}}
                    for probs in prediction_probs
                ]]

            @api(input=JsonInput(), batch=True)
            def predictactive(self, parsed_json_list: List[JsonSerializable]):
                text = []
                n_instances = 1
                for json in parsed_json_list:
                    if 'batch' in json:
                        text = json['batch']
                    if 'n_instances' in json:
                        n_instances = json['n_instances']
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')

                np_indices, res_text_list = self.artifacts.active_learning_function(
                    text,
                    n_instances,
                    self.artifacts.function,
                    self.artifacts.model
                )

                return [(np_indices.tolist(), res_text_list)]


        '''

transformers_string = '''
        @artifacts([{0}('tokenizer_model'), PickleArtifact('function'), PickleArtifact('active_learning_function')])
        class TemplateModel(BentoService):

            @api(input=JsonInput(), batch=True)
            def predict(self, parsed_json_list: List[JsonSerializable]):
            
                model = self.artifacts.tokenizer_model.get("model")
                tokenizer = self.artifacts.tokenizer_model.get("tokenizer")
                
                text = []
                for json in parsed_json_list:
                    if 'text' in json:
                        text.append(json['text'])
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')

                prediction_probs, class_names, _ = self.artifacts.function(
                    model,
                    tokenizer,
                    text
                )

                return [
                    {{class_names[i]: prob for i,
                        prob in enumerate(probs)}}
                    for probs in prediction_probs
                ]

            @api(input=JsonInput(), batch=True)
            def predictbatch(self, parsed_json_list: List[JsonSerializable]):
            
                model = self.artifacts.tokenizer_model.get("model")
                tokenizer = self.artifacts.tokenizer_model.get("tokenizer")
                
                text = []
                for json in parsed_json_list:
                    if 'batch' in json:
                        text = json['batch']
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')
                prediction_probs, class_names, _ = self.artifacts.function(
                    model,
                    tokenizer,
                    text
                )

                return [[
                    {{class_names[i]: prob for i,
                        prob in enumerate(probs)}}
                    for probs in prediction_probs
                ]]

            @api(input=JsonInput(), batch=True)
            def predictactive(self, parsed_json_list: List[JsonSerializable]):
            
                model = self.artifacts.tokenizer_model.get("model")
                tokenizer = self.artifacts.tokenizer_model.get("tokenizer")
                
                text = []
                n_instances = 1
                for json in parsed_json_list:
                    if 'batch' in json:
                        text = json['batch']
                    if 'n_instances' in json:
                        n_instances = json['n_instances']
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')

                np_indices, res_text_list = self.artifacts.active_learning_function(
                    text,
                    n_instances,
                    self.artifacts.function,
                    model,
                    tokenizer,
                )

                return [(np_indices.tolist(), res_text_list)]
        '''

tokenizer_string = '''
        @artifacts([{0}('model'), PickleArtifact('tokenizer'), PickleArtifact('function'), PickleArtifact('active_learning_function')])
        class TemplateModel(BentoService):

            @api(input=JsonInput(), batch=True)
            def predict(self, parsed_json_list: List[JsonSerializable]):

                model = self.artifacts.model
                tokenizer = self.artifacts.tokenizer

                text = []
                for json in parsed_json_list:
                    if 'text' in json:
                        text.append(json['text'])
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')

                prediction_probs, class_names, _ = self.artifacts.function(
                    model,
                    tokenizer,
                    text
                )

                return [
                    {{class_names[i]: prob for i,
                        prob in enumerate(probs)}}
                    for probs in prediction_probs
                ]

            @api(input=JsonInput(), batch=True)
            def predictbatch(self, parsed_json_list: List[JsonSerializable]):
                
                model = self.artifacts.model
                tokenizer = self.artifacts.tokenizer
                
                text = []
                for json in parsed_json_list:
                    if 'batch' in json:
                        text = json['batch']
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')
                prediction_probs, class_names, _ = self.artifacts.function(
                    model,
                    tokenizer,
                    text
                )

                return [[
                    {{class_names[i]: prob for i,
                        prob in enumerate(probs)}}
                    for probs in prediction_probs
                ]]

            @api(input=JsonInput(), batch=True)
            def predictactive(self, parsed_json_list: List[JsonSerializable]):
            
                model = self.artifacts.model
                tokenizer = self.artifacts.tokenizer
                
                text = []
                n_instances = 1
                for json in parsed_json_list:
                    if 'batch' in json:
                        text = json['batch']
                    if 'n_instances' in json:
                        n_instances = json['n_instances']
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')

                np_indices, res_text_list = self.artifacts.active_learning_function(
                    text,
                    n_instances,
                    self.artifacts.function,
                    model,
                    tokenizer,
                )

                return [(np_indices.tolist(), res_text_list)]
        '''
