#ifndef _H_AGG_H
#define _H_AGG_H

#include <vector>
#include <set>


class Agg
{
    public:
        //generators
        void generate(int _n, int _N);
        void generate_from_file(int _n, int _N, char* file);
        
        void generate_outlier(int _n, int _N);
        void generate_2norm(int _n, int _N);
        void generate_gamma(int _n, int _N);
        
        //aggregators
        //ip-mu
        std::pair<double,std::vector<std::vector<double> > > aggregate_ip2_warmstart(int K);
        
        //ip-lambda
        std::pair<double,std::vector<std::vector<double> > > aggregate_ip_lambda2(int K);
        
        //cont
        std::pair<double,std::vector<std::vector<double> > > aggregate_restart(int K, int R);
        
        //k means
        std::vector<std::vector<double> > aggregate_kmeans(int K, int R);
        
        
        //aux functions
        double evaluation(std::vector<double> x, std::vector<std::vector<double> > scen);
        std::vector<std::vector<double> > get_c();
        
        std::vector<double> solve_selection(std::vector<std::vector<double> > scen, int p);
        std::vector<double> solve_vertexcover(std::vector<std::vector<double> > scen, std::vector<std::pair<int,int> > edges);
        
        void set_timelimit(double tl);
        
    private:
        int n,N;
        
        std::vector<std::vector<double> > c;
    
        double distance(std::vector<double> vec1, std::vector<double> vec2);
        
        double timelimit;
};
    
#endif
