only_model_string = '''
        @artifacts([{0}('model'), PickleArtifact('function')])
        class {1}(BentoService):
        
            @api(input=JsonInput())
            def predict(self, parsed_json: JsonSerializable):
                text = parsed_json['text']
                
                return self.artifacts.function(
                    self.artifacts.model,
                    text
                )

            @api(input=JsonInput())
            def predictactive(self,  parsed_json: JsonSerializable):
                text = parsed_json['text']
                n_instances = parsed_json['n_instances']

                np_indices, res_text_list = active_learning_function(
                    text,
                    n_instances,
                    self.artifacts.function,
                    self.artifacts.model
                )

                return [(np_indices.tolist(), res_text_list)]

        '''

transformers_string = '''
        @artifacts([{0}('model'), PickleArtifact('function'), PickleArtifact('active_learning_function')])
        class {1}(BentoService):
        
            @api(input=JsonInput())
            def predict(self, parsed_json: JsonSerializable):
                text = parsed_json['text']
                model = self.artifacts.model.get("model")
                tokenizer = self.artifacts.model.get("tokenizer")
                
                return self.artifacts.function(
                    model,
                    tokenizer,
                    text
                )

            @api(input=JsonInput())
            def predictactive(self, parsed_json: JsonSerializable):
                text = parsed_json['text']
                n_instances = parsed_json['n_instances']
                model = self.artifacts.model.get("model")
                tokenizer = self.artifacts.model.get("tokenizer")

                np_indices, res_text_list = active_learning_function(
                    text,
                    n_instances,
                    self.artifacts.function,
                    model,
                    tokenizer,
                )

                return [(np_indices.tolist(), res_text_list)]
        '''

tokenizer_and_vocab_string = '''
        @artifacts([{0}('model'), PickleArtifact('tokenizer'), PickleArtifact('vocab'), PickleArtifact('function'), PickleArtifact('active_learning_function')])
        class {1}(BentoService):
        
            @api(input=JsonInput())
            def predict(self, parsed_json: JsonSerializable):
                text = parsed_json['text']
                model = self.artifacts.model
                tokenizer = self.artifacts.tokenizer
                vocab = self.artifacts.vocab
                
                return self.artifacts.function(
                    model,
                    tokenizer,
                    vocab,
                    text
                )

            @api(input=JsonInput())
            def predictactive(self, parsed_json: JsonSerializable):
                text = parsed_json['text']
                n_instances = parsed_json['n_instances']
                model = self.artifacts.model
                tokenizer = self.artifacts.tokenizer
                vocab = self.artifacts.vocab

                np_indices, res_text_list = active_learning_function(
                    text,
                    n_instances,
                    self.artifacts.function,
                    model,
                    tokenizer,
                    vocab
                )

                return [(np_indices.tolist(), res_text_list)]
        '''
