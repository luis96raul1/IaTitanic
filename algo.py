import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
#df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df=pd.read_csv("./titanic.csv")
print(df.describe())
df.drop(['Siblings/Spouses','Parents/Children'],axis=1,inplace=True)
print(df.head())
keys=['Survived','Pclass','Sex','Age','Fare'] #2 sex
caracteristicas=keys[1:len(keys)]
etiqueta=keys[0]
#print(df[caracteristicas])
#print(df[etiqueta])
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]
def train_input_fn(traindf):
	return tf.compat.v1.estimator.inputs.pandas_input_fn(
		x=traindf[caracteristicas],
		y=traindf[etiqueta],
		batch_size=40,
		num_epochs=500,
		shuffle=True,
		queue_capacity=1000
		)
def eval_input_fn(evaldf):
	return tf.compat.v1.estimator.inputs.pandas_input_fn(
		x=evaldf[caracteristicas],
		y=evaldf[etiqueta],
		batch_size=40,
		shuffle=False,
		queue_capacity=1000
		)
def predict_input_fn(newdf):
	return tf.compat.v1.estimator.inputs.pandas_input_fn(
		x=newdf,
		y=None,
		batch_size=10,
		shuffle=False)
categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key='Sex',vocabulary_list=('male','female'))#Indicator columns and embedding columns never work on features directly
featuress=[tf.feature_column.numeric_column('Pclass'),
		   tf.feature_column.numeric_column('Fare'),
		   tf.feature_column.numeric_column('Age'),
		   tf.feature_column.indicator_column(categorical_column)]
outdir='titanic_trained2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
#shutil.rmtree(outdir, ignore_errors = True) # start fresh each time
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
model=tf.estimator.DNNRegressor(hidden_units=[15,10,5],feature_columns=featuress,model_dir=outdir,activation_fn=(tf.nn.sigmoid),optimizer=opt)
model.train(train_input_fn(df))

def print_rmse(model, df):
  metrics = model.evaluate(input_fn = eval_input_fn(evaldf))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

print_rmse(model,evaldf)
tf.estimator.RunConfig(model_dir=outdir)
#ejecutar en CMD 
#python -m tensorboard.main --logdir=titanic_trained2

while True:
	s=input("Género (male/female) :")
	a=input("Clase (1,2,3): ")
	b=input("Edad: (:80): ")
	c=input("Tarifa (0:40): ")
	newdf=pd.DataFrame({'Pclass':[int(a)],'Age':[int(b)],'Sex':[str(s)],'Fare':[int(c)]})
	prediccion=model.predict(predict_input_fn(newdf))
	print(next(prediccion))
	s=str(input("salir? [y/n]:"))
	if s=="y":
		break;