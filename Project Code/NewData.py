import pyreadstat

#Replication Data for: 'Double for Nothing? Experimental Evidence on an Unconditional Teacher Salary Increase in Indonesia'
#https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910/DVN/MTVM5O


# Access the variable names
variable_names = meta.column_names
print("Variables in the file:")
print(variable_names)
# Using pyreadstat
columns_to_drop=['teacher_id','triplet_id', 'district_id','school_id','interviewed','SD','quota',
                 'PAID','TARGET', 'NOTELIGIBLE','S1', 'pursue','ngscore','certification_pay','quota_year','year', 'tested']
df = df.drop(columns=columns_to_drop )
df=df.dropna()

df2, meta = pyreadstat.read_dta('C:/Users/geminia/Desktop/Project Code/students.dta')
df2= df2[['assets', 'ITT_score']]
df2=df2.dropna()
df2 = df2.iloc[0:5469,:]

df['assets'] = df2['assets']
df['ITT_score'] = df2['ITT_score']

df=df.dropna()
df.rename(columns={'treatment': 't'}, inplace=True)
df = df.astype(float) #fondamentale se no non funziona 

# X | t
continuous_vars_0 = ["totalpay", "assets",'base_pay','additional_pay']
continuous_lower_bounds_0 = { "totalpay": 0, "assets":0,"base_pay" : 0,"additional_pay" : 0}
categorical_vars_0 = ["secondjob", "secondjobhours", "problems", "happy","absent","certified"]
context_vars_0 = ["t"] #This is what you are conditioning on, that is in the first case only on treatment 

# Y | X, t
continuous_vars_1 = ["ITT_score"] #This is your Y: Real earnings in 1978 
continuous_lower_bounds_1 = {"ITT_score": 0}
categorical_vars_1 = [] #You don't want to generate any categorical variables in this case 
context_vars_1 = ["t","secondjob", "totalpay",'base_pay','additional_pay',"secondjobhours", "problems", "assets","happy","absent","certified"]


min_val = df['ITT_score'].min()
max_val = df['ITT_score'].max()
df['ITT_score'] = (df['ITT_score'] - min_val) / (max_val - min_val)

rows_to_drop = df.sample(n=2000, random_state=42).index
df = df.drop(index=rows_to_drop)

df_balanced = df.sample(2 * len(df),weights=(1 - df.t.mean()) * df.t + df.t.mean() * (1 - df.t),replace=True,)


compare_dfs(df, df_generated,scatterplot=dict( x=["base_pay","totalpay","additional_pay"], y=["ITT_score", "assets"], samples=400, smooth=0), 
            table_groupby=["t"], histogram=dict(variables=["ITT_score", "totalpay", "happy", "secondjobhours","assets","absent"], nrow=3, ncol=2),figsize=3)

#200 batch size, the df size is 