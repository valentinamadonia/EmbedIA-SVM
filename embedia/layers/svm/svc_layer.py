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
        char * kernel_type = "{self.layer.kernel.lower()}";
        int degree = {self.layer.degree};
        double gamma = {self.layer.gamma};
        double  coef0 = {self.layer.coef0};
        double C = {self.layer.C};
        double rho[] = {'{' + ', '.join(map(str, self.layer.intercept_)) + '}'};
        int nSV[] = {'{' + ', '.join(map(str, self.layer.n_support_)) + '}'};
        double SV[][] = {{'''
        for vector in self.layer.support_vectors_:
            init_svc_layer += f'        {{' + ', '.join(map(str, vector)) + '}},\n'
        init_svc_layer += '''    }};
        double dual_coef[][] = {{ '''
        for row in self.layer.dual_coef_:
            init_svc_layer += f'        {{' + ', '.join(map(str, row)) + '}},\n'
        
        init_svc_layer += f'''    }};

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