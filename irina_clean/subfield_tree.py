class SubfieldTree():

    def __init__(self, df_subfield):
        self.df_subfield = df_subfield.sort_values(["domain_id", "field_id", "subfield_id"])

    @property
    def domain_list(self):
        return self.df_subfield.domain_id.unique().tolist()
    
    @property
    def domain_name_list(self):
        return self.df_subfield.groupby("domain_id").domain_name.first().loc[self.domain_list].tolist()
    
    @property
    def field_list(self):
        return self.df_subfield.field_id.unique().tolist()
    
    @property
    def field_name_list(self):
        return self.df_subfield.groupby("field_id").field_name.first().loc[self.field_list].tolist()
    
    @property
    def ordered_subfield_ids(self):
        return self.df_subfield.subfield_id.tolist()
    
    @property
    def field_name_dict(self):
        return self.df_subfield.set_index("subfield_id").field_name.to_dict()
    
    @property
    def domain_name_dict(self):
        return self.df_subfield.set_index("subfield_id").domain_name.to_dict()
    
    @property
    def field_domain_name_dict(self):
        return self.df_subfield.set_index("field_name").domain_name.to_dict()