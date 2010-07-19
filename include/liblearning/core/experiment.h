#ifndef EXPERIMENT_H_
#define EXPERIMENT_H_

#include "parameter_set.h"
#include "experiment_datasets.h"

#include <algorithm>
#include <string>
#include <fstream>
#include <tuple>
using namespace std;

#include <boost/filesystem.hpp> 
#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>

#include <boost/multi_array.hpp>



template<typename M,typename FP>
class experiment
{

protected:

	string experiment_name;

    const experiment_datasets & datasets;
	
    parameter_set   train_candidate;
    
	parameter_set   test_candidate;

	FP fixed_param;

	ofstream logfile;

public:
	experiment(const experiment_datasets & datasets_, const string & experiment_name_):datasets(datasets_), experiment_name(experiment_name_)
	{
		//if ( boost::filesystem::exists( experiment_name ))
		//{
		//	throw "the log file with same name :" + experiment_name + " has already exist!";
		//}

		logfile.open(experiment_name.c_str());
	}
	~experiment(void)
	{
		logfile.close();
	}

public:

	virtual tuple<shared_ptr<dataset>, shared_ptr<dataset>> prepare_dataset(const dataset & train, const dataset & test) = 0 ;

	virtual M train_one_machine(const dataset & train, const vector<double> & train_params) = 0 ;

    virtual double test_performance(M & ,const dataset & train, const dataset & test, const vector<double> &  test_params) = 0 ;

	virtual void load_config(const string & config_file) = 0;

public:

	void set_fixed_param(const FP & param)
	{
		fixed_param = param;
	}

	void set_param_candidate(const parameter_set & train_candidate_, const parameter_set & test_candidate_)
	{
		train_candidate = train_candidate_;
		test_candidate = test_candidate_;
	}


private:

	void log_experiment_configuration()
	{
		logfile << " The configuration of experiment : " << endl;
		logfile << " Training-testing pairs : " << datasets.get_train_test_pair_num() << endl;
		logfile << " Cross-validation folder nums for each training set : " ;
		for (int i = 0;i < datasets.get_train_test_pair_num();i++)
		{
			logfile << datasets.get_train_test_pair(i).get_cv_folder_num() << " " ;
		}
		logfile << endl << endl;

		logfile << " There are " << train_candidate.get_param_num() <<  " training parameter candidates :" << endl ;
		for (int i = 0; i < train_candidate.get_param_num(); i ++ )
		{
			logfile << "/t the candidates for " << i <<  " -th training parameter are :" ;
			const vector<double> & cur_train_candidate = train_candidate.get_param_candidate(i);
			for_each(cur_train_candidate.begin(),cur_train_candidate.end(),[this](double param){logfile << param << " " ;});
			logfile << endl;
		}
		logfile << endl << endl;

		logfile << " There are " << test_candidate.get_param_num() <<  " test parameter candidates :" << endl ;
		for (int i = 0; i < test_candidate.get_param_num(); i ++ )
		{
			logfile << "/t the candidates for " << i <<  " -th testing parameter are :" ;
			const vector<double> & cur_test_candidate = test_candidate.get_param_candidate(i);
			for_each(cur_test_candidate.begin(),cur_test_candidate.end(),[this](double param){logfile << param << " " ;});
			logfile << endl;
		}
		logfile << endl << endl;

	}

	tuple<vector<double>, vector<double>,double > select_param_one_pair(const vector<cross_validation_pair > & cv_pairs)
	{
            
        vector<vector<double>> train_param_comb = train_candidate.emurate_parameter_combination();
		vector<vector<double>> test_param_comb = test_candidate.emurate_parameter_combination();

		MatrixXd performance = MatrixXd::Zero(train_param_comb.size(),test_param_comb.size());

		boost::thread_group train_test_threads;

		boost::mutex performance_update_mutex;

        for (int m = 0; m < train_param_comb.size(); m++)
        {
			train_test_threads.create_thread([&, m ](){
				for (int j = 0; j < cv_pairs.size(); j++)
				{

						shared_ptr<dataset>  train = cv_pairs[j].get_train_dataset();
						shared_ptr<dataset>  valid = cv_pairs[j].get_validation_dataset();

						shared_ptr<dataset>  proc_train, proc_valid;
						tie(proc_train,proc_valid) = prepare_dataset(*train,*valid);

						M machine = train_one_machine(*proc_train,train_param_comb[m]);
				
						for (int k = 0; k < test_param_comb.size(); k++)
						{
							double cur_perf  = test_performance(machine, *proc_train, *proc_valid, test_param_comb[k]);

							{
								boost::mutex::scoped_lock
								lock(performance_update_mutex);
								performance(m,k) += cur_perf;
							}
						}

						std::cout<< "Finish evaluating one training parameters:" << m << " at " << j << " -th cv-pairs" << endl;
				}
			});

		}

		train_test_threads.join_all();

		auto abs_max_pos = max_element(performance.data(), performance.data()+performance.size());
		int relative_max_pos = abs_max_pos - performance.data();

		// performance (in type of MaxtrixXd) is column major
		int row = relative_max_pos% performance.rows();
		int col = relative_max_pos/ performance.rows();

		double validation_performance = performance(row,col)/cv_pairs.size();

		return make_tuple(train_param_comb[row],test_param_comb[col],validation_performance);

	}

	double calculate_test_performance(const dataset & train, const dataset & test, const vector<double> & optim_train_param, const vector<double> & optim_test_param)
	{
		M machine = train_one_machine(train,optim_train_param);
        return  test_performance( machine,train,test, optim_test_param);
	}

public:

    double evaluate(int iter_num)
	{
		log_experiment_configuration();

		logfile << "Begining experiment: " << endl;

		double over_all_perf = 0;

		for (int i = 0; i < iter_num ; i ++)
		{
			logfile << "------------------------------------------------" << endl;

			logfile << "The results for " << i <<"-th experiment: " << endl;

			double perf = 0;

			for(int  j = 0; j < datasets.get_train_test_pair_num(); j++)
			{

				vector<double> optim_train_params, optim_test_params;
				double cur_valid_perf;
				tie(optim_train_params, optim_test_params,cur_valid_perf) = select_param_one_pair(datasets.get_train_test_pair(j).get_all_cv_pairs());

				shared_ptr<dataset> train = datasets.get_train_test_pair(j).get_train_dataset();
				shared_ptr<dataset> test = datasets.get_train_test_pair(j).get_test_dataset();

				shared_ptr<dataset> proc_train, proc_test;
				tie(proc_train,proc_test) = prepare_dataset(*train,*test);

				double cur_test_perf = calculate_test_performance(*proc_train, *proc_test,  optim_train_params, optim_test_params);
            
				logfile << "/t" << "The experimental result for " << j <<"-th train-test pair: " <<endl;
				logfile << "/t" << "/t The optimal training params are : " ;
				for_each(optim_train_params.begin(),optim_train_params.end(),[this](double param){logfile << param << " " ;});
				logfile << endl;
				logfile << "/t" << "/t The optimal testing params are : " ;
				for_each(optim_test_params.begin(),optim_test_params.end(),[this](double param){logfile << param << " " ;});
				logfile << endl;
				logfile << "/t" << "/t The best average validation performance is : " << cur_valid_perf << endl; 
				logfile << "/t" << "/t The test performance is : " << cur_test_perf << endl; 
				logfile << endl;

				logfile.flush();

               
				perf = perf + cur_test_perf;

				
			}

           
			perf = perf/datasets.get_train_test_pair_num();

			logfile << "/t" << "The testing performance for " << i << "-th experiment is : " << perf << endl;

			over_all_perf += perf;
		}

		logfile << "The average testing performance for all experiments is : " << over_all_perf << endl;

		logfile << "The experiment is finished." << endl;

		return over_all_perf;

	}
};

#endif
