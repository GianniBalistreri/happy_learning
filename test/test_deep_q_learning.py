import os
import pandas as pd
import unittest

from happy_learning.deep_q_learning import DQNAgent
from happy_learning.sampler import MLSampler
from happy_learning.supervised_machine_learning import ModelGeneratorClf
from typing import List

OUTPUT_PATH: str = './data/'
DF: pd.DataFrame = pd.read_csv(filepath_or_buffer=f'{OUTPUT_PATH}avocado.csv', sep=',')
DF = DF.replace({'conventional': 0, 'organic': 1})
DF['type'] = DF['type'].astype(int)
CLF_TARGET: str = 'type'
REG_TARGET: str = 'AveragePrice'
FEATURES: List[str] = ['Total Volume',
                       'Total Bags',
                       '4046',
                       '4225',
                       '4770',
                       'Total Bags',
                       'Small Bags',
                       'Large Bags',
                       'XLarge Bags'
                       ]


class TestDQNAgent(unittest.TestCase):
   """
   Class for testing methods of class DQNAgent
   """
   def test_apply_learning(self):
       """
       Test apply learning (inference) function
       """
       _model_name: str = 'cat'
       _param: dict = ModelGeneratorClf(model_name=_model_name).generate_model().model_param
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm=_model_name)
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       _new_param: dict = agent.apply_learning(model_name=_model_name,
                                               param=_param,
                                               file_path_policy_network=f'{OUTPUT_PATH}rl_policy_net.p',
                                               file_path_environment=f'{OUTPUT_PATH}rl_env.p',
                                               file_path_model=f'{OUTPUT_PATH}model.p',
                                               data_sets=None,
                                               max_actions=1,
                                               force_all_actions=False
                                               )
       self.assertNotEqual(first=_param, second=_new_param)

   def test_optimize_clf_ada(self):
       """
       Test hyper-parameter optimization of ada boost classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='ada')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_cat(self):
       """
       Test hyper-parameter optimization of cat boost classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='cat')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_lida(self):
       """
       Test hyper-parameter optimization of linear discriminant analysis classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='lida')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_log(self):
       """
       Test hyper-parameter optimization of logistic regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='log')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_gbo(self):
       """
       Test hyper-parameter optimization of gradient boosting decision tree classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='gbo')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_knn(self):
       """
       Test hyper-parameter optimization of k-nearest neighbor classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='knn')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_qda(self):
       """
       Test hyper-parameter optimization of quadratic discriminant analysis classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='qda')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_rf(self):
       """
       Test hyper-parameter optimization of random forest classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='rf')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_svm(self):
       """
       Test hyper-parameter optimization of support vector machine classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='svm')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_nusvm(self):
       """
       Test hyper-parameter optimization of nu support vector machine classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='nusvm')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_clf_xgb(self):
       """
       Test hyper-parameter optimization of extreme gradient boosting decision tree classification using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='xgb')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_ada(self):
       """
       Test hyper-parameter optimization of ada boost regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='ada')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_cat(self):
       """
       Test hyper-parameter optimization of cat boost regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='cat')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_elastic(self):
       """
       Test hyper-parameter optimization of elastic net regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='elastic')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_gam(self):
       """
       Test hyper-parameter optimization of generalized additive models regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='gam')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_rf(self):
       """
       Test hyper-parameter optimization of random forest regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='rf')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_svm(self):
       """
       Test hyper-parameter optimization of support vector machine regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='svm')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_nusvm(self):
       """
       Test hyper-parameter optimization of nu support vector machine regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='nusvm')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_gbo(self):
       """
       Test hyper-parameter optimization of gradient boosting decision tree regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='gbo')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_lasso(self):
       """
       Test hyper-parameter optimization of lasso regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='lasso')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_knn(self):
       """
       Test hyper-parameter optimization of k-nearest neighbor regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='knn')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_reg_xgb(self):
       """
       Test hyper-parameter optimization of extreme gradient boosting decision tree regression using reinforcement learning (Deep-Q-Learning)
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='xgb')
                                  )
       _n_optimization_steps_before: int = agent.n_optimization
       agent.optimize(df=DF, target=REG_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_before)

   def test_optimize_continue(self):
       """
       Test continuing optimization
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='cat')
                                  )
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       _n_optimization_steps_after: int = agent.n_optimization
       agent.optimize_continue(df=DF, target=CLF_TARGET, features=FEATURES)
       self.assertGreater(a=agent.n_optimization, b=_n_optimization_steps_after)

   def test_save(self):
       """
       Test save function
       """
       data_sets: dict = MLSampler(df=DF,
                                   target=CLF_TARGET,
                                   features=FEATURES,
                                   train_size=0.8,
                                   stratification=False
                                   ).train_test_sampling(validation_split=0.1)
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='cat')
                                  )
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       agent.save(agent=True, model=True, data_sets=data_sets, experience=True)
       _found_saved_files: int = 0
       if os.path.isfile(f'{OUTPUT_PATH}rl_agent.p'):
           _found_saved_files += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_env.p'):
           _found_saved_files += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_policy_net.p'):
           _found_saved_files += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_target_net.p'):
           _found_saved_files += 1
       if os.path.isfile(f'{OUTPUT_PATH}model.p'):
           _found_saved_files += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_agent_experience.p'):
           _found_saved_files += 1
       self.assertTrue(expr=_found_saved_files == 6)

   def test_visualize(self):
       """
       Test visualization function
       """
       agent: DQNAgent = DQNAgent(episodes=10,
                                  target_update=5,
                                  output_file_path=OUTPUT_PATH,
                                  **dict(sml_algorithm='cat')
                                  )
       agent.optimize(df=DF, target=CLF_TARGET, features=FEATURES)
       agent.visualize()
       _found_saved_plots: int = 0
       if os.path.isfile(f'{OUTPUT_PATH}rl_metadata_table.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_experience_metadata.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_experience_time_step.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_reward_distribution.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_reward_distribution_per_episode.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_reward_score_joint_distribution.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_sml_score_distribution.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_sml_score_distribution_per_episode.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_reward_by_action_learning_type.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_action_distribution.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_action_learning_type_distribution.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_action_distribution_grouped_by_action_type.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_network_loss.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_network_loss_per_episode.html'):
           _found_saved_plots += 1
       if os.path.isfile(f'{OUTPUT_PATH}rl_action_learning_type_distribution.html'):
           _found_saved_plots += 1
       self.assertTrue(expr=_found_saved_plots >= 12)


if __name__ == '__main__':
    unittest.main()
