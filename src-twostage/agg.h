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
        
        
        //aggregators
        std::pair<std::vector<std::vector<double> >, std::vector<std::vector<double> > > aggregate_ip2(int K);
        
        std::vector<std::vector<double> > aggregate_kmeans(int K, int R);
        
        
        //aux functions
        std::vector<double> solve_selection(std::vector<std::vector<double> > scen, int p);
        double evaluation_selection(std::vector<double> x, std::vector<std::vector<double> > scen, int p);
        
        std::vector<double> solve_vertexcover(std::vector<std::vector<double> > scen, std::vector<std::pair<int,int> > edges);
        double evaluation_vertexcover(std::vector<double> x, std::vector<std::vector<double> > scen, std::vector<std::pair<int,int> > edges);
        
        std::vector<std::vector<double> > get_c();
        
        void set_timelimit(double tl);
        
    private:
        int n,N;
        
        std::vector<double> C;
        std::vector<std::vector<double> > c;
    
        double distance(std::vector<double> vec1, std::vector<double> vec2);
        
        double timelimit;
};
    
#endif
