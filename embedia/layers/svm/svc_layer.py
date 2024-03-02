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
        uint16_t i,j;
        uint16_t nr_class = {self.layer.classes_.size};
        uint16_t nr_SV = {len(self.layer.support_)};

        uint16_t *label;
        label = malloc( sizeof(uint16_t) * nr_class);
        uint16_t copied_label[] = {'{' + ', '.join(map(str, self.layer.classes_)) + '}'};
        memcpy(label,copied_label,sizeof(uint16_t) * nr_class);

        char * kernel_type = "{self.layer.kernel.lower()}";
        uint16_t degree = {self.layer.degree};
        float gamma = {self.layer.gamma};
        float  coef0 = {self.layer.coef0};

        float *rho;
        rho = malloc(sizeof(float) * {self.layer.intercept_.size});
        float copied_rho[] = {'{' + ', '.join(map(str, self.layer.intercept_)) + '}'};
        memcpy(rho,copied_rho,sizeof(float) * {self.layer.intercept_.size});

        uint16_t *nSV;
        nSV = malloc(sizeof(uint16_t) * {self.layer.n_support_.size});
        uint16_t copied_nSV[] = {'{' + ', '.join(map(str, self.layer.n_support_)) + '}'};
        memcpy(nSV,copied_nSV,sizeof(uint16_t) * {self.layer.n_support_.size});

        float **SV;
        SV = malloc( sizeof(float*) * nr_SV);
        for(i= 0 ; i<nr_SV; i++) SV[i]= malloc( sizeof(float) * { self.layer.support_vectors_[0].size});
        '''
        for i in range(len(self.layer.support_)):  
            init_svc_layer += f'    memcpy(SV[{i}], (float[]) {{' + ', '.join(map(str,self.layer.support_vectors_[i])) + f'}},sizeof(float) * { self.layer.support_vectors_[0].size});\n'
        
        init_svc_layer += f'''
        float **dual_coef;
        dual_coef = malloc( sizeof(float*) * nr_class - 1);
        for(i= 0 ; i<nr_class - 1; i++) dual_coef[i]= malloc( sizeof(float) * nr_SV);
        '''
        for i in range(self.layer.classes_.size - 1):  
            init_svc_layer += f'    memcpy(dual_coef[{i}], (float[]) {{' + ', '.join(map(str,self.layer.dual_coef_[i])) + f'}},sizeof(float) * nr_SV);\n'

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
    	free(s_v_c_data.label);
    	free(s_v_c_data.rho);
    	free(s_v_c_data.nSV);

            return layer;
        }}
        '''

        return init_svc_layer
    
    def predict(self, input_name, output_name):
        return f'''svc_layer({self.name}_data, {input_name}, &{output_name});'''