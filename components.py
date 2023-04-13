import kfp
import kfp.components as comp
import requests
import kfp.dsl as dsl
import functions

create_step_prepare_data = kfp.components.create_component_from_func(
    func=functions.prepare_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4', 'numpy==1.21.0']
)


create_step_train_test_split = kfp.components.create_component_from_func(
    func=functions.train_test_split,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4', 'numpy==1.21.0', 'scikit-learn==0.24.2']
)


create_step_training_basic_classifier = kfp.components.create_component_from_func(
    func=functions.training_basic_classifier,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4', 'numpy==1.21.0', 'scikit-learn==0.24.2']
)


create_step_predict_on_test_data = kfp.components.create_component_from_func(
    func=functions.predict_on_test_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4', 'numpy==1.21.0', 'scikit-learn==0.24.2']
)


create_step_predict_prob_on_test_data = kfp.components.create_component_from_func(
    func=functions.predict_prob_on_test_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4', 'numpy==1.21.0', 'scikit-learn==0.24.2']
)


create_step_get_metrics = kfp.components.create_component_from_func(
    func=functions.get_metrics,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4', 'numpy==1.21.0', 'scikit-learn==0.24.2']
)

##############################################################################################
# Define the pipeline
@dsl.pipeline(
    name='IRIS classifier Kubeflow Demo Pipeline',
    description='A sample pipeline that performs IRIS classifier task'
)
# Define parameters to be fed into pipeline
def iris_classifier_pipeline(data_path: str):
    vop = dsl.VolumeOp(
    name ="t-vol",
    resource_name ="t-vol",
    size ="1Gi",
    modes =dsl.VOLUME_MODE_RWO)

    prepare_data_task = create_step_prepare_data().add_pvolumes({data_path: vop.volume})
    train_test_split = create_step_train_test_split().add_pvolumes({data_path: vop.volume}).after(prepare_data_task)
    classifier_training = create_step_training_basic_classifier().add_pvolumes({data_path: vop.volume}).after(
        train_test_split)
    log_predicted_class = create_step_predict_on_test_data().add_pvolumes({data_path: vop.volume}).after(
        classifier_training)
    log_predicted_probabilities = create_step_predict_prob_on_test_data().add_pvolumes({data_path: vop.volume}).after(
        log_predicted_class)
    log_metrics_task = create_step_get_metrics().add_pvolumes({data_path: vop.volume}).after(
        log_predicted_probabilities)

    prepare_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_test_split.execution_options.caching_strategy.max_cache_staleness = "P0D"
    classifier_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_class.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_probabilities.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_metrics_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

###########################################################################################
kfp.compiler.Compiler().compile(
    pipeline_func=iris_classifier_pipeline,
    package_path='IRIS_Classifier_pipeline1.yaml')


#########################################################################################
KUBEFLOW_URI = "https://qa.unifytwin.com/kubeflow_admin/"
LOGIN_TOKEN = "5f719768f0154a0b9a32ba1cdbfea09d"

#test


##############################################################################################
DATA_PATH = '/data'

import datetime
print(datetime.datetime.now().date())


pipeline_func = iris_classifier_pipeline
experiment_name = 'iris_classifier_exp' +"_"+ str(datetime.datetime.now().date())
run_name = pipeline_func.__name__ + ' run'
namespace = "kubeflow"

arguments = {"data_path":DATA_PATH}

kfp.compiler.Compiler().compile(pipeline_func,
  '{}.zip'.format(experiment_name))

run_result = client.create_run_from_pipeline_func(pipeline_func,
                                                  experiment_name=experiment_name,
                                                  run_name=run_name,
                                                  arguments=arguments)



