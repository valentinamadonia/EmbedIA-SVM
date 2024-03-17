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
        uint16_t i;
        uint16_t nr_class = {self.layer.classes_.size};
        uint16_t nr_SV = {len(self.layer.support_)};

        static uint16_t label[] = {'{' + ', '.join(map(str, self.layer.classes_)) + '}'};

        char * kernel_type = "{self.layer.kernel.lower()}";
        uint16_t degree = {self.layer.degree};
        float gamma = {self.layer.gamma};
        float  coef0 = {self.layer.coef0};

        static float rho[] = {'{' + ', '.join(map(str, self.layer.intercept_)) + '}'};

        static uint16_t nSV[] = {'{' + ', '.join(map(str, self.layer.n_support_)) + '}'};

        static float * SV[{len(self.layer.support_)}];
        '''
        for i in range(len(self.layer.support_)):  
            init_svc_layer += f'   static float SV{i}[] = {{' + ', '.join(map(str,self.layer.support_vectors_[i])) + f'}};\n'
            init_svc_layer += f'   SV[{i}] = SV{i};'
        
        init_svc_layer += f'''
        static float *dual_coef[{self.layer.classes_.size - 1}];
        '''
        for i in range(self.layer.classes_.size - 1):  
            init_svc_layer += f'   static float d_coef{i}[] ={{' + ', '.join(map(str,self.layer.dual_coef_[i])) + f'}};\n'
            init_svc_layer += f'   dual_coef[{i}] = d_coef{i};'
        init_svc_layer += f'''
        svc_layer_t layer = {{
                nr_class,
                nr_SV,
                kernel_type,
                degree,
                gamma,
                coef0,
                label,
                rho,
                nSV,
                SV,
                dual_coef
        }};

        for (i = 0; i < s_v_c_data.nr_SV; i++) {{
        	free(s_v_c_data.SV[i]);
    	}}
    	free(s_v_c_data.SV);
    	for (i = 0; i < s_v_c_data.nr_class -1; i++) {{
        	free(s_v_c_data.dual_coef[i]);
    	}}
    	free(s_v_c_data.dual_coef);

            return layer;
        }}
        '''

        return init_svc_layer
    
    def predict(self, input_name, output_name):
        return f'''svc_layer({self.name}_data, {input_name}, &{output_name});'''