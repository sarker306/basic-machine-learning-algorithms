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

using namespace std;

typedef long long ll;
typedef unsigned long long llu;

class SequentialLearner{
private:
    vector< vector<string> > data;
    vector<string> attrs;
public:
    SequentialLearner(vector< vector<string> > _data);
    vector< vector< pair<string, string> > > learnRules(){}
};

SequentialLearner::SequentialLearner(vector< vector< string > > _data) : data(_data){
    if ( data.size() > 0 ){
        string curAttr;
        for ( int i = 0 ; i < data[0].size() ; i++ ){
            if ( curAttr.size() == 0 || curAttr[curAttr.size() - 1] == 'Z' )
                curAttr += 'A';
            else curAttr[curAttr.size() - 1]++;
            attrs.push_back(curAttr);
        }
    }

    /*for ( int i = 0 ; i < attrs.size() ; i++ ) cout << attrs[i] << " ";
    cout << endl;
    for ( int i = 0 ; i < data.size() ; i++ ){
        cout << data[i].size() << " ";
        for ( int j = 0 ; j < data[i].size() ; j++ ) cout << data[i][j] << " ";
        cout << endl;
    }*/
}

vector< vector< pair<string, string> > > SequentialLearner::learnRules(){
    vector < vector< pair<string, string> > > rules;
    
    if ( data.size() == 0 ) return rules;
    
    
}

vector<vector<string> > inputData(){
    vector<vector<string> > data;
    freopen("data.csv", "r", stdin);
    
	for ( int i = 0 ; ; i++ ) {
		vector<string> now;
        char word[100];
        int num;

		if ( scanf("%d", &num) != 1 ) break;
        sprintf(word, "%d", num);
        now.push_back(word);
		for ( int j = 1 ; j < 10 ; j++ ){
            scanf(",%d", &num);
            sprintf(word, "%d", num);
            now.push_back(word);
        }

        data.push_back(now);
	}

	return data;
}

int main(int argc, char **argv){
    if ( argc != 1 ) return 1;
    SequentialLearner s(inputData());
    
    return 0;
}
