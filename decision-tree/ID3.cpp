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

#define FOR(i, a, b) for (i = (a); i <= (b); i++)
#define REP(i, a) for (i = 0; i < (a); i++)
#define ALL(v) (v).begin(), (v).end()
#define SET(a, x) memset((a), (x), sizeof(a))
#define SZ(a) ((int)(a).size())
#define CL(a) ((a).clear())
#define SORT(x) sort(ALL(x))
#define mp make_pair
#define pb push_back

#define filer() freopen("in.txt","r",stdin)
#define filew() freopen("out.txt","w",stdout)
#define MAX_ATTR 9
#define TARGET_DOMAIN_LEN 2

using namespace std;

typedef long long ll;
typedef unsigned long long llu;

struct Node{
	int label;
	int attr;
	Node* child[11];

	Node(int l = -1){
		label = l;
		attr = -1;
		for ( int i = 0 ; i < 11  ; i++ ) child[i] = 0;
	}
};

struct Tree{
	Node *root;
};

struct Sample{
	int attr[MAX_ATTR];
	int res;

	void print(){
		for ( int i = 0 ; i < 9 ; i++ ) printf("%2d ", attr[i]);
		printf("%2d\n", res);
	}
};

void clearNode(Node *root);
void inputData();
void createTrainData(vector<int>&);

vector<Sample> data;
vector<int> dataInd;
int trainData = 0;

Node* ID3(vector<int> &arr, unsigned attrMask);
int predict(Node *root, Sample &smp);
double entropy(vector<int> &arr);
int findBestAttr(vector<int> &arr, unsigned attrMask);
double misClassificationImpurity(vector<int>&, int);
double informationGain(vector<int>&, int);
double (*fPtr)(vector<int>&, int);
double avgFnmr, avgFmr, avgAccr;

void train(Tree &tree){
	double fnmr = 0, fmr = 0, accr = 0;
	vector<int> trainInds;

 	createTrainData(trainInds);

	tree.root = ID3(trainInds, 0);

	// testing phase
	for ( int i = trainData ; i < dataInd.size() ; i++ ){
		int testRes = predict(tree.root, data[dataInd[i]]);
		if ( testRes == 0 && data[dataInd[i]].res == 1 ) fnmr++;
		if ( testRes == 1 && data[dataInd[i]].res == 0 ) fmr++;
		else accr++;
	}
	
    accr *= 100.0 / (dataInd.size() - trainData);
	fnmr /= (dataInd.size() - trainData);
	fmr /= (dataInd.size() - trainData);
	//printf("Fnmr = %lf, Fmr = %lf\n", fnmr, fmr);
	avgFmr += fmr, avgFnmr += fnmr, avgAccr += accr;
}

int main(int argc, char **argv){
	Tree tree;
	int run = 1000;

	ios::sync_with_stdio(true);

	/*if ( argc != 2 ){
		printf("Usage : ID3 datapath\n");
		return 0;
	}

	freopen(argv[1], "r", stdin);*/
	freopen("E:\\Virtual_Desktop\\Study_3_2\\Machine Learning\\Assignments\\data.csv", "r", stdin);
	inputData();

	fPtr = misClassificationImpurity;
	avgFmr = avgFnmr = avgAccr = 0;
	for ( int i = 0 ; i < run ; i++ ){
		tree.root = 0;
		train(tree);
		clearNode(tree.root);
	}
	printf("Average fnmr = %lf, Average fmr = %lf, Average accr = %lf\n", avgFnmr / run, avgFmr / run, avgAccr / run);

	fPtr = informationGain;
	avgFmr = avgFnmr = avgAccr = 0;
	for ( int i = 0 ; i < run ; i++ ){
		tree.root = 0;
		train(tree);
		clearNode(tree.root);
	}
	printf("Average fnmr = %lf, Average fmr = %lf, Average accr = %lf\n", avgFnmr / run, avgFmr / run, avgAccr / run);

	return 0;
}

/* Decision Tree Methods */
void clearNode(Node *root){
	if ( root == 0 ) return;
	for ( int i = 0 ; i < 10 ; i++ )
		clearNode(root->child[i]);

	delete root;
}

/* Data input */
void inputData(){
	for ( int i = 0 ; ; i++ ) {
		Sample now;
		if ( scanf("%d", &now.attr[0]) != 1 ) break;
		for ( int j = 1 ; j < 9 ; j++ ) scanf(",%d", &now.attr[j]);
		scanf(",%d", &now.res);
		data.pb(now);
		dataInd.pb(i);
	}

	trainData = (int)(data.size() * 0.8);
}

/* Data preprocessing */
void createTrainData(vector<int> &arr){
	random_shuffle(dataInd.begin(), dataInd.end());
	copy(dataInd.begin(), dataInd.begin() + trainData, back_inserter(arr));
}

/* Training */
double entropy(vector<int> &arr){
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

double informationGain(vector<int> &arr, int a){
	double gain = entropy(arr);

	for ( int i = 1 ; i <= 10 ; i++ ){
		vector<int> temp;
		for ( int j = 0 ; j < arr.size() ; j++ )
			if ( data[arr[j]].attr[a] == i ) temp.pb(arr[j]);

		if ( temp.size() > 0 ) gain -= ((temp.size() * 1.0)/arr.size()) * entropy(temp);
	}

	return gain;
}

double misClassificationImpurity(vector<int> &arr, int a){
	double cnt[11], maxProb = 0;

	memset(cnt, 0, sizeof(cnt));
	for ( int i = 0 ; i < arr.size() ; i++ ) cnt[data[arr[i]].attr[a]]++;
	for ( int i = 0 ; i < 11 ; i++ ) if ( maxProb < cnt[i] ) maxProb = cnt[i];
	maxProb /= arr.size();

	return 1 - maxProb;
}

int findBestAttr(vector<int> &arr, unsigned attrMask){
	int best = -1;
	double maxGain = -1e20;
	double initialEntropy = entropy(arr);

	for ( int i = 0 ; i < MAX_ATTR ; i++ ){
		if ( attrMask & (1<<i) ) continue;

		double gain = (*fPtr)(arr, i);
		if ( gain > maxGain ) maxGain = gain, best = i;
	}

	return best;
}

int mostCommonValueTargetAttr(vector<int> &arr){
	int cnt[TARGET_DOMAIN_LEN], i;

	for ( i = 0 ; i < TARGET_DOMAIN_LEN ; i++ ) cnt[i] = 0;
	for ( i = 0 ; i < arr.size() ; i++ ) cnt[data[arr[i]].res]++;

	int maxCnt = 0, maxCntInd = -1;
	for ( i = 0 ; i < TARGET_DOMAIN_LEN; i++ )
		if ( cnt[i] > maxCnt ) maxCnt = cnt[i], maxCntInd = i;

	return maxCntInd;
}

Node* ID3(vector<int> &arr, unsigned attrMask){
	vector<int> partition[11];
	int i;

	int res = -1;
	if ( arr.size() > 0 ) res = data[arr[0]].res;
	for ( i = 0 ; i < arr.size() ; i++ ) if ( data[arr[i]].res != res ) break;
	if ( i == arr.size() ){
		return new Node(res);
	}

	if ( attrMask == (1U << MAX_ATTR) - 1 ){
		return new Node(mostCommonValueTargetAttr(arr));
	}

	int bestAttr = findBestAttr(arr, attrMask);
	Node* nowNode = new Node();
	nowNode->attr = bestAttr;

	for ( i = 0 ; i < arr.size() ; i++ )
		partition[data[arr[i]].attr[bestAttr]].pb(arr[i]);

	for ( i = 1 ; i <= 10 ; i++ ){
		if ( partition[i].size() == 0 )
			nowNode->child[i] = new Node(mostCommonValueTargetAttr(arr));
		else nowNode->child[i] = ID3(partition[i], attrMask | (1<<bestAttr));
	}

	return nowNode;
}

int predict(Node *root, Sample &smp){
	if ( root->label != -1 ) return root->label;

	return predict(root->child[smp.attr[root->attr]], smp);
}
