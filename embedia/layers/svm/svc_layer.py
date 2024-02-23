from embedia.layers.data_layer import DataLayer
from embedia.model_generator.project_options import ModelDataType

class SVC_layer(DataLayer):
        
    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)
        self.struct_data_type = 'svc_layer_t'
    
    def functions_init(self):

        struct_type = self.struct_data_type
        name = self.name
    
        init_svc_layer = f'''
        {struct_type} init_{name}_data(void){{
        char * kernel_type = "{self.model.kernel.lower()}";
        int degree = {self.model.degree};
        double gamma = {self.model.gamma};
        double  coef0 = {self.model.coef0};
        double C = {self.model.C};
        double rho[] = {'{' + ', '.join(map(str, self.model.intercept_)) + '}'};
        int nSV[] = {'{' + ', '.join(map(str, self.model.n_support_)) + '}'};
        double SV[][] = {{'''
        for vector in self.model.support_vectors_:
            init_svm_layer += f'        {'{' + ', '.join(map(str, vector)) + '}'},\n'
        init_svm_layer += '''    }};
        double dual_coef[][] = {{ '''
        for row in self.model.dual_coef_:
            init_svm_layer += f'        {'{' + ', '.join(map(str, row)) + '}'},\n'
        
        init_svm_layer += f'''    }};

        svc_layer_t layer = {{
                kernel_type,
                degree,
                gamma,
                coef0,
                C,
                {{{{sizeof(rho) / sizeof(rho[0])}}, rho}},
                {{{{sizeof(nSV) / sizeof(nSV[0])}}, nSV}},
                {{{{sizeof(SV) / sizeof(SV[0])}}, SV}},
                {{{{sizeof(dual_coef) / sizeof(dual_coef[0])}}, dual_coef}}
        }};
            return layer;
        }}
        '''
        return init_svc_layer
    
    def predict(self, input_name, output_name):
        return f'''svc_layer({self.name}_data, {input_name}, &{output_name});'''