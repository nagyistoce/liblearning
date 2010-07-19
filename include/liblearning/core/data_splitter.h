#ifndef BATCH_MAKER
#define	BATCH_MAKER


#include <vector>
using namespace std;

class dataset;
class dataset_splitter
{
public:
	dataset_splitter(void);
	~dataset_splitter(void);

	virtual vector<vector<int>> split(const dataset& data, int batch_num) const  = 0;
};


class ordered_dataset_splitter :public dataset_splitter
{
public:
	ordered_dataset_splitter();
	virtual vector<vector<int>> split(const dataset& data,int batch_num) const;
};

class random_shuffer_dataset_splitter : public dataset_splitter
{
public:
	random_shuffer_dataset_splitter();
	virtual vector<vector<int>> split(const dataset& data,int batch_num) const;
};

class supervised_random_shuffer_dataset_splitter : public dataset_splitter
{

public:
	supervised_random_shuffer_dataset_splitter();
	virtual vector<vector<int>> split(const dataset& data,int batch_num) const ;
};

#endif

