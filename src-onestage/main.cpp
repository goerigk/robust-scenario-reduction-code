#include "agg.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
    
    srand(atoi(argv[5]));
    
    int n = atoi(argv[1]);
    int p = n/2;
    int N = atoi(argv[2]);
    double timelimit = atoi(argv[3]);
    int useopt = atoi(argv[4]);
    
    Agg agg;
    agg.generate(n, N);        
    agg.set_timelimit(timelimit);
    
    vector<pair<int,int> > edges;
    for (int i=0; i<n; ++i)
        for (int j=i+1; j<n; ++j)
            if (rand()%1000 < 10.0*1000.0/n)
                edges.push_back(make_pair(i,j));
    
    cout<<edges.size()<<"\n"<<flush;
    
    double start = clock();
    double vcopt = agg.evaluation(agg.solve_vertexcover(agg.get_c(),edges),agg.get_c());
    cout<<"VCOPTTIME = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
    
    start = clock();
    double selopt = agg.evaluation(agg.solve_selection(agg.get_c(),p),agg.get_c());
    cout<<"SELOPTTIME = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
    
    for (int K=1; K<11; K+=1)
    {
        start = clock();
        pair<double,vector<vector<double> > > retagg2 = agg.aggregate_restart(K,10);
        cout<<K<<";AGGTIME2 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";

        start = clock();
        vector<double> vcx2 = agg.solve_vertexcover(retagg2.second,edges);
        cout<<K<<";VCSOLVETIME2 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
        cout<<K<<";VCUBRATIO2 = "<<agg.evaluation(vcx2,agg.get_c())/vcopt<<"\n";
        
        start = clock();
        vector<double> selx2 = agg.solve_selection(retagg2.second,p);
        cout<<K<<";SELSOLVETIME2 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
        cout<<K<<";SELUBRATIO2 = "<<agg.evaluation(selx2,agg.get_c())/selopt<<"\n";
        
        
        start = clock();
        vector<vector<double> > retagg3 = agg.aggregate_kmeans(K,1000);
        cout<<K<<";AGGTIME3 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
        
        start = clock();
        vector<double> vcx3 = agg.solve_vertexcover(retagg3,edges);
        cout<<K<<";VCSOLVETIME3 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
        cout<<K<<";VCUBRATIO3 = "<<agg.evaluation(vcx3,agg.get_c())/vcopt<<"\n";
        
        start = clock();
        vector<double> selx3 = agg.solve_selection(retagg3,p);
        cout<<K<<";SELSOLVETIME3 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
        cout<<K<<";SELUBRATIO3 = "<<agg.evaluation(selx3,agg.get_c())/selopt<<"\n";


        if (useopt > 0.5)
        {
            start = clock();
            pair<double,vector<vector<double> > > retagg1 = agg.aggregate_ip2_warmstart(K);
            cout<<K<<";AGGTIME1 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";

            start = clock();
            vector<double> vcx1 = agg.solve_vertexcover(retagg1.second,edges);
            cout<<K<<";VCSOLVETIME1 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
            cout<<K<<";VCUBRATIO1 = "<<agg.evaluation(vcx1,agg.get_c())/vcopt<<"\n";
            
            start = clock();
            vector<double> selx1 = agg.solve_selection(retagg1.second,p);
            cout<<K<<";SELSOLVETIME1 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
            cout<<K<<";SELUBRATIO1 = "<<agg.evaluation(selx1,agg.get_c())/selopt<<"\n";
            
        
            start = clock();
            pair<double,vector<vector<double> > > retagg4 = agg.aggregate_ip_lambda2(K);
            cout<<K<<";AGGTIME4 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";

            start = clock();
            vector<double> vcx4 = agg.solve_vertexcover(retagg4.second,edges);
            cout<<K<<";VCSOLVETIME4 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
            cout<<K<<";VCUBRATIO4 = "<<agg.evaluation(vcx4,agg.get_c())/vcopt<<"\n";
            
            start = clock();
            vector<double> selx4 = agg.solve_selection(retagg4.second,p);
            cout<<K<<";SELSOLVETIME4 = "<<(clock()-start)/CLOCKS_PER_SEC<<"\n";
            cout<<K<<";SELUBRATIO4 = "<<agg.evaluation(selx4,agg.get_c())/selopt<<"\n";
            
            
            cout<<K<<";guarantees;"<<retagg1.first<<";"<<retagg2.first<<";"<<retagg4.first<<"\n";

        }
        

    }
    
    return 0;
}



