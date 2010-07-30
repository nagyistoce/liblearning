
#include <liblearning/deeplearning/objective/fisher_objective.h>

using namespace Eigen;

#include <liblearning/core/supervised_dataset.h>
#include <liblearning/deeplearning/deep_auto_encoder.h>



fisher_objective::fisher_objective()
{
	type = encoder_related;

}

fisher_objective::~fisher_objective()
{
}



void fisher_objective::set_dataset(const dataset & data_set_)
{
	if (data_set == &data_set_)
		return ;

	data_set = &data_set_;

	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(*data_set);

	int sample_num = s_data_set.get_sample_num();
	int class_num = s_data_set.get_class_num();

	const vector<int> & label = s_data_set.get_label();
	const vector<int> & class_id = s_data_set.get_class_id();
	const vector<int> & class_elem_num = s_data_set.get_class_elem_num();

	Aw.resize(sample_num,sample_num);
	Ab.resize(sample_num,sample_num);

    int i, j,k;

    for (i = 0;i<sample_num;i++)
    {
        for (j = 0;j<sample_num;j++)
        {
            if (label[i] == label[j])
            {
                for (k = 0;k<class_num;k++)
                {
                    if (class_id[k] == label[i])
                        break;
                }
                double nc = class_elem_num[k];
                Aw(i,j) = 1.0/nc;
                Ab(i,j) = 1.0/sample_num - 1/nc;
            }
            else
            {
                Aw(i,j) = 0;
                Ab(i,j) = 1.0/sample_num;
            }
        }
    }

	Aw_diff_helper = Aw + Aw.transpose();
	VectorXd Aw_diag = Aw_diff_helper.colwise().sum();
	Aw_diag.asDiagonal().subTo(Aw_diff_helper);

	Ab_diff_helper = Ab + Ab.transpose();
	VectorXd Ab_diag = Ab_diff_helper.colwise().sum();
	Ab_diag.asDiagonal().subTo(Ab_diff_helper);
}

#include <liblearning/util/Eigen_util.h>

#include <limits>

double fisher_objective::prepared_value(deep_auto_encoder & net) 
{
	const MatrixXd & feature = net.get_layered_output(net.get_encoder_layer_id());

	MatrixXd M = sqdist(feature,feature);
           
	trSw = 0.5*(Aw.array()*M.array()).sum();// + std::numeric_limits<double>::epsilon();
    trSb = 0.5*(Ab.array()*M.array()).sum();// + std::numeric_limits<double>::epsilon();
    double value = trSw / trSb;

	return value;

}


vector<shared_ptr<MatrixXd>> fisher_objective::prepared_value_diff(deep_auto_encoder & net) 
{
	const MatrixXd & feature = net.get_layered_output(net.get_encoder_layer_id());

	//MatrixXd Aw_AwT = Aw + Aw.transpose();
	//VectorXd Aw_diag = Aw_AwT.colwise().sum();
	//Aw_diag.asDiagonal().subTo(Aw_AwT);
	MatrixXd JSw = -feature*Aw_diff_helper;  

	//MatrixXd Ab_AbT = Ab + Ab.transpose();
	//VectorXd Ab_diag = Ab_AbT.colwise().sum();
	//Aw_diag.asDiagonal().subTo(Aw_AwT);
    MatrixXd JSb = -feature*Ab_diff_helper;  

    shared_ptr<MatrixXd> JF (new MatrixXd((JSw-JSb*(trSw/trSb))/trSb)); 

	vector<shared_ptr<MatrixXd>> result(2);

	result[0] = JF ;
	result[1] = shared_ptr<MatrixXd>();
	return result;
}


fisher_objective * fisher_objective::clone()
{
	return new fisher_objective(*this);
}