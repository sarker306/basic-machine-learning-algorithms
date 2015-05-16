#include <cstdio>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <ctime>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <deque>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <bitset>

#define FOR(i, a, b) for (i = (a); i <= (b); i++)
#define REP(i, a) for (i = 0; i < (a); i++)
#define ALL(v) (v).begin(), (v).end()
#define SET(a, x) memset((a), (x), sizeof(a))
#define SZ(a) ((int)(a).size())
#define CL(a) ((a).clear())
#define SORT(x) sort(ALL(x))
#define mp make_pair
#define pb push_back
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

#define filer() freopen("in.txt","r",stdin)
#define filew() freopen("out.txt","w",stdout)
#define MAX_ATTR 9
#define MAX_ATTR_VAL 10
#define TARGET_DOMAIN_LEN 2

using namespace std;

typedef long long ll;
typedef unsigned long long llu;

struct Sample{
	int attr[MAX_ATTR];
	int res;

	void print(){
		for ( int i = 0 ; i < 9 ; i++ ) printf("%2d ", attr[i]);
		printf("%2d\n", res);
	}
};

class Learner{
public:
    virtual int predict(Sample &smp) = 0;
};

class Stump : public Learner{
public:
    int label[MAX_ATTR_VAL + 1];
    int attr;
    
    Stump(){
        memset(label, 0, sizeof(label));
        attr = -1;
    }
    
    int predict(Sample &smp){
        return label[smp.attr[attr]];
    }
};

double getRandom(){
	double now = rand();
	return now / RAND_MAX;
}

double entropy(vector<Sample> &data, vector<int> &arr){
	double res = 0;
	int cnt[TARGET_DOMAIN_LEN];

	for ( int i = 0 ; i < TARGET_DOMAIN_LEN ; i++ ) cnt[i] = 0;

	for ( int i = 0 ; i < arr.size() ; i++ ) cnt[data[arr[i]].res]++;
	for ( int i = 0 ; i < TARGET_DOMAIN_LEN ; i++ ){
		double tmp = (cnt[i] * 1.0) / arr.size();
		if ( fabs(tmp) > 1e-6 )	res += -tmp * log(tmp);
	}

	return res;
}

double informationGain(vector<Sample> &data, vector<int> &arr, int a){
	double gain = entropy(data, arr);

	for ( int i = 1 ; i <= MAX_ATTR_VAL ; i++ ){
		vector<int> temp;
		for ( int j = 0 ; j < arr.size() ; j++ )
			if ( data[arr[j]].attr[a] == i ) temp.pb(arr[j]);

		if ( temp.size() > 0 ) gain -= ((temp.size() * 1.0)/arr.size()) * entropy(data, temp);
	}

	return gain;
}

int findBestAttr(vector<Sample> &data, vector<int> &arr){
	int best = -1;
	double maxGain = -1e200;

	for ( int i = 0 ; i < MAX_ATTR ; i++ ){
        double gain = informationGain(data, arr, i);
		if ( gain > maxGain ) maxGain = gain, best = i;
	}

	return best;
}

Learner* learn(vector<Sample> &data, vector<double> &prob){
    vector<double> cumulated(prob.size());
    vector<int> dataInd(data.size());
    vector<int> partition[MAX_ATTR_VAL + 1];
    
    for ( int i = 0 ; i < prob.size() ; i++ ) cumulated[i] = prob[i];
    for ( int i = 1 ; i < prob.size() ; i++ ) cumulated[i] += cumulated[i-1];
    
   // for ( int i = 0 ; i < prob.size() ; i++ ) printf("%d(%.3lf) ", i, cumulated[i]);
   // puts("");
    
    for ( int i = 0 ; i < data.size() ; i++ ){
        double nextVal = getRandom();
    //    printf("%.3lf", nextVal);
        dataInd[i] = lower_bound(cumulated.begin(), cumulated.end(), nextVal) - cumulated.begin();
    //    printf("(%d) ", dataInd[i]);
    }
   // puts("");

    Stump* stump = new Stump();
    stump->attr = findBestAttr(data, dataInd);

	for ( int i = 0 ; i < dataInd.size() ; i++ )
		partition[data[dataInd[i]].attr[stump->attr]].push_back(dataInd[i]);

    for ( int i = 1 ; i <= MAX_ATTR_VAL ; i++ ){
        vector<int> cnt(TARGET_DOMAIN_LEN);
        int maxVal = -100;
        
        for ( int j = 0 ; j < partition[i].size() ; j++ )
            cnt[data[partition[i][j]].res]++;

        for ( int j = 0 ; j < TARGET_DOMAIN_LEN ; j++ )
            if ( maxVal < cnt[j] ) maxVal = cnt[j], stump->label[i] = j;
    }
    
    return stump;
}

void cleanUpLearners(vector<Learner*> learners){
    for ( int i = 0 ; i < learners.size() ; i++ ) delete learners[i];
}

void AdaBoost(vector<Sample> &data, int k){
    vector<double> prob(data.size()), weight(data.size());
    vector<bool> isPredicted(data.size());
    vector<double>beta(k);
    vector<Learner*> learners(k);
    
    for ( int i = 0 ; i < data.size() ; i++ ) weight[i] = 1.0 / data.size();
    
    for ( int r = 0 ; r < k ; r++ ){
        double sumWgt = 0;
        for ( int i = 0 ; i < data.size() ; i++ ) sumWgt += weight[i];
        for ( int i = 0 ; i < data.size() ; i++ ) prob[i] = weight[i] / sumWgt;
        
        /* learn */
        learners[r] = learn(data, prob);
        /* test */
        double errVal = 0;
        double accr = 0;
        
        for ( int i = 0 ; i < data.size() ; i++ )
            if ( learners[r]->predict(data[i]) != data[i].res ) errVal += prob[i], isPredicted[i] = 0;
            else isPredicted[i] = 1, accr++;

        
        if ( errVal > 0.5 ){
            printf("ErrVal more than 0.5, returning\n");
            cleanUpLearners(learners);
            return;
        }
        
        beta[r] = errVal / ( 1 - errVal );
        accr *= 100.0 / data.size();
        //printf("ErrVal[%d] = %lf, beta = %lf, accr = %lf\n", r, errVal, beta[r], accr);
        
        for ( int i = 0 ; i < data.size() ; i++ )
            if ( isPredicted[i] == 1 ) weight[i] = weight[i] * beta[r];
    }
    
    double accr = 0;
    for ( int i = 0 ; i < data.size() ; i++ ){
        vector<int> cnt(TARGET_DOMAIN_LEN);
        
        for ( int r = 0 ; r < k ; r++ ) cnt[learners[r]->predict(data[i])]++;
        int maxVal = -1, maxInd;
        
        for ( int j = 0 ; j < TARGET_DOMAIN_LEN ; j++ )
            if ( maxVal < cnt[j] ) maxVal = cnt[j], maxInd = j;

        if ( maxInd == data[i].res ) accr++;
    }
    
    accr *= 100.0 / data.size();
    printf("Accuracy : %lf\n", accr);
    
    /* cleanup */
    cleanUpLearners(learners);
    printf("Done");
}

/* Data input */
void inputData(vector<Sample> &data){
	for ( int i = 0 ; ; i++ ) {
		Sample now;
		if ( scanf("%d", &now.attr[0]) != 1 ) break;
		for ( int j = 1 ; j < 9 ; j++ ) scanf(",%d", &now.attr[j]);
		scanf(",%d", &now.res);
		data.push_back(now);
	}
	
	printf("Data size : %d\n", data.size());
}

int main(int argc, char** argv){
    vector<Sample> data;
    int k = 50;
    
    freopen(argv[1], "r", stdin);
    k = atoi(argv[2]);
	inputData(data);
	AdaBoost(data, k);

    return 0;
}
