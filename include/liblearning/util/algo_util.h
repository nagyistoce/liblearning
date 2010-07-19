#ifndef ALGO_UTIL_H_
#define ALGO_UTIL_H_


#include <algorithm>
#include <iterator>
#include <vector>
#include <boost/iterator/counting_iterator.hpp>




template <class RandomAccessIterator>
vector<unsigned int> sort_index(RandomAccessIterator first, RandomAccessIterator last)
{
	int N = last-first;
	vector<unsigned int> index(N);

	std::copy(
		boost::counting_iterator<unsigned int>(0),
		boost::counting_iterator<unsigned int>(N), 
		std::back_inserter(index));

	std::sort(index.begin(),index.end(),
		[&](const unsigned int a, const unsigned int b) ->bool 
		{ 
			return * (first + a) < * (first + b);
		}
	);

	return index;

}

template <class RandomAccessIterator>
vector<unsigned int> nth_element_index(
	RandomAccessIterator first, 
	RandomAccessIterator middle, 
	RandomAccessIterator last
	)
{
	int N = last-first;
	int nth_pos = middle - first;
	vector<unsigned int> index(N);

	std::copy(
		boost::counting_iterator<unsigned int>(0),
		boost::counting_iterator<unsigned int>(N), 
		std::back_inserter(index));

	std::nth_element(index.begin(),index.begin()+nth_pos,index.end(),
		[&](const unsigned int a, const unsigned int b) ->bool 
		{ 
			return * (first + a) < * (first + b);
		}
	);

	index.erase(index.begin()+nth_pos,index.end());

	return index;

}

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


template <typename T>
vector<T> construct_array(const string & coded_str)
{
	typedef vector< string > split_vector_type;

	split_vector_type v;
	boost::split( v, coded_str, boost::is_space() );
	
	auto v_end = remove_if(v.begin(),v.end(),[](const string & e)-> bool { return e.empty(); });

	int num = v_end - v.begin();

	vector<T > t_v(num);

	for (int i = 0;i < num ;i++)
	{
		t_v[i] = boost::lexical_cast<T>(v[i]);
	}

	return t_v;

}



#endif
