#include "agg.h"
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "ilcplex/ilocplex.h"

ILOSTLBEGIN


using namespace std;


void Agg::generate(int _n, int _N)
{
    n = _n;
    N = _N;

    C.resize(n);
    for (int j=0; j<n; ++j)
        C[j] = rand()%100+1;
            
	c.resize(N);
    for (int i=0; i<N; ++i)
	{
		c[i].resize(n);
		for (int j=0; j<n; ++j)
            c[i][j] = rand()%100+1;
	}
}

void Agg::generate_from_file(int _n, int _N, char* file)
{
    n = _n;
    N = _N;
    
    ifstream in(file);
    string line;
    
    while (!in.eof())
	{
		getline(in,line);
		
		// Ignore Comment and Empty Lines
		if (line=="" || line[0]=='#')
			continue;

        vector<double> scen(n);

        scen[0] = atof(line.c_str());
        for (int j=1; j<n; ++j)
        {
            size_t pos = line.find(";");
            line=line.substr(pos+1);
            scen[j] = atof(line.c_str());
            scen[j] = min(100.0,scen[j]);
            scen[j] = max(1.0,scen[j]);
        }
        c.push_back(scen);
	}
	
	in.close();
}

vector<vector<double> > Agg::get_c()
{
    return c;
}


void Agg::set_timelimit(double tl)
{
    timelimit = tl;
}

vector<vector<double> > Agg::aggregate_kmeans(int K, int R)
{
    double bestdist = N*n*100;
    vector<vector<double> > bestc;
    
    for (int rep=0; rep<R; ++rep)
    {
    
        vector<vector<double> > centre(K);
        vector<int> available(N);
        for (int i=0; i<N; ++i)
            available[i] = i;
        random_shuffle(available.begin(), available.end());
        for (int k=0; k<K; ++k)
            centre[k] = c[available[k]];

        double curdist;
        double newdist = N*n*100;
        
        do
        {     
            curdist = newdist;
            newdist = 0;
            
            vector<vector<double> > currentmu(N);
            
            for (int i=0; i<N; ++i)
                currentmu[i].resize(K,0);

            for (int i=0; i<N; ++i)
            {
                int mink = 0;
                double mindist = distance(c[i],centre[0]);
                for (int k=1; k<K; ++k)
                {
                    double dist = distance(c[i],centre[k]);
                    if (dist < mindist)
                    {
                        mink = k;
                        mindist = dist;
                    }
                }
                currentmu[i][mink] = 1;
                newdist += mindist;
            }
            
            centre.clear();
            centre.resize(K);
            
            for (int k=0; k<K; ++k)
            {
                vector<int> assigned;
                for (int i=0; i<N; ++i)
                    if (currentmu[i][k] > 0.5)
                        assigned.push_back(i);
                
                double S = assigned.size();
                centre[k].resize(n,0);
                for (int i=0; i<S; ++i)
                    for (int j=0; j<n; ++j)                
                        centre[k][j] += c[assigned[i]][j]/S;
            }

        } while (newdist + 0.1 < curdist); 
    
        if (newdist < bestdist)
        {
            bestdist = newdist;
            bestc = centre;
        }
    
    }
    
    return bestc;
}


vector<double> Agg::solve_selection(vector<vector<double> > scen, int p)
{
    //local N
    int N = scen.size();
    
    IloEnv env;
	IloModel model(env);

	vector<IloNumVar> cplexx(n);
	for (int j=0; j<n; ++j)
		cplexx[j] = IloNumVar(env, 0, 1, ILOBOOL);
        
    vector<vector<IloNumVar> > cplexy(N);
    for (int i=0; i<N; ++i)
    {
        cplexy[i].resize(n);
        for (int j=0; j<n; ++j)
            cplexy[i][j] = IloNumVar(env, 0, 1, ILOBOOL);
    }
	
    IloNumVar cplext(env,0,IloInfinity,ILOFLOAT);
    
    for (int i=0; i<N; ++i)
	{ 
		IloExpr con(env);
		for (int j=0; j<n; ++j)
            con += scen[i][j] * cplexy[i][j];
		
		model.add(con<=cplext);
	}
    
    for (int i=0; i<N; ++i)
    {
        IloExpr con(env);
        for (int j=0; j<n; ++j)
            con += cplexx[j] + cplexy[i][j];
        model.add(con == p);
    }
    
    for (int i=0; i<N; ++i)
        for (int j=0; j<n; ++j)
            model.add(cplexx[j] + cplexy[i][j] <= 1);
    
    IloExpr obj(env);
    for (int j=0; j<n; ++j)
        obj += C[j]*cplexx[j];
    model.add(IloMinimize(env,obj + cplext));

	IloCplex cplex(model);

	cplex.setOut(env.getNullStream());
	cplex.setParam(IloCplex::Threads,1);
    if (timelimit > 0)
        cplex.setParam(IloCplex::TiLim, timelimit);

	bool result = cplex.solve();

    vector<double> x(n,0);
    for (int j=0; j<n; ++j)
        if (cplex.getValue(cplexx[j]) > 0.5)
            x[j] = 1;

    env.end();

    return x;

}


double Agg::distance(vector<double> vec1, vector<double> vec2)
{
    double sqrsum = 0;

    for (int j=0; j<n; ++j)
        sqrsum += (vec1[j] - vec2[j]) * (vec1[j] - vec2[j]);

    return sqrt(sqrsum);
}

double Agg::evaluation_selection(vector<double> x, vector<vector<double> > scen, int p)
{
    int N = scen.size();
    int remp = p;
    double firstcost = 0;
    for (int j=0; j<n; ++j)
        if (x[j] > 0.5)
        {
            firstcost += C[j];
            remp--;
        }
    
    double maxscen = 0;
    for (int i=0; i<N; ++i)
    {
        double obj = 0;
        vector<double> vals;
        for (int j=0; j<n; ++j)
            if (x[j] < 0.5)
                vals.push_back(scen[i][j]);
        sort(vals.begin(), vals.end());
        for (int r=0; r<remp; ++r)
            obj += vals[r];
        maxscen = max(maxscen, obj);
    }
    
	return firstcost + maxscen;
}

vector<double> Agg::solve_vertexcover(vector<vector<double> > scen, vector<pair<int,int> > edges)
{
    //local N
    int N = scen.size();
    
    IloEnv env;
	IloModel model(env);

	vector<IloNumVar> cplexx(n);
	for (int j=0; j<n; ++j)
		cplexx[j] = IloNumVar(env, 0, 1, ILOBOOL);
        
    vector<vector<IloNumVar> > cplexy(N);
	for (int i=0; i<N; ++i)
	{
        cplexy[i].resize(n);
        for (int j=0; j<n; ++j)
            cplexy[i][j] = IloNumVar(env, 0, 1, ILOBOOL);
    }
    
    IloNumVar cplext(env,0,IloInfinity,ILOFLOAT);
    
    for (int i=0; i<N; ++i)
	{ 
		IloExpr con(env);
		for (int j=0; j<n; ++j)
            con += scen[i][j] * cplexy[i][j];
		
		model.add(con<=cplext);
	}
    
    
    for (int i=0; i<N; ++i)
        for (int j=0; j<n; ++j)
        {
            IloExpr con(env);
            for (int l=0; l<edges.size(); ++l)
            {
                if (edges[l].first == j)
                    con += cplexx[edges[l].second] + cplexy[i][edges[l].second];
                if (edges[l].second == j)
                    con += cplexx[edges[l].first] + cplexy[i][edges[l].first];
            }
            con += cplexx[j] + cplexy[i][j];
            model.add(con >= 1);
        }
        
    
    for (int i=0; i<N; ++i)
        for (int j=0; j<n; ++j)
            model.add(cplexx[j] + cplexy[i][j] <= 1);
    
    IloExpr obj(env);
    for (int j=0; j<n; ++j)
        obj += C[j]*cplexx[j];
    model.add(IloMinimize(env,obj + cplext));

	IloCplex cplex(model);

	cplex.setOut(env.getNullStream());
	cplex.setParam(IloCplex::Threads,1);
    if (timelimit > 0)
        cplex.setParam(IloCplex::TiLim, timelimit);

	bool result = cplex.solve();

    vector<double> x(n,0);
    for (int j=0; j<n; ++j)
        if (cplex.getValue(cplexx[j]) > 0.5)
            x[j] = 1;

    env.end();

    return x;   
}


double Agg::evaluation_vertexcover(vector<double> x, vector<vector<double> > scen, vector<pair<int,int> > edges)
{
    int N = scen.size();
    double firstcosts = 0;
    for (int j=0; j<n; ++j)
        if (x[j] > 0.5)
            firstcosts += C[j];
            
    double wc = 0;
    for (int i=0; i<N; ++i)
    {
        IloEnv env;
        IloModel model(env);

        vector<IloNumVar> cplexy(n);
        for (int j=0; j<n; ++j)
            cplexy[j] = IloNumVar(env, 0, 1, ILOBOOL);
    
    
        for (int j=0; j<n; ++j)
        {
            IloExpr con(env);
            for (int l=0; l<edges.size(); ++l)
            {
                if (edges[l].first == j)
                    con += x[edges[l].second] + cplexy[edges[l].second];
                if (edges[l].second == j)
                    con += x[edges[l].first] + cplexy[edges[l].first];
            }
            con += x[j] + cplexy[j];
            model.add(con >= 1);
        }
        
    
        for (int j=0; j<n; ++j)
            model.add(x[j] + cplexy[j] <= 1);
    
        IloExpr obj(env);
        for (int j=0; j<n; ++j)
            obj += scen[i][j]*cplexy[j];
        model.add(IloMinimize(env,obj));

        IloCplex cplex(model);

        cplex.setOut(env.getNullStream());
        cplex.setParam(IloCplex::Threads,1);
        
        bool result = cplex.solve();

        double val = cplex.getObjValue();
        
        env.end();
        
        wc = max(wc, val);
    }
    
    return firstcosts + wc;
}




pair<vector<vector<double> >, vector<vector<double> > > Agg::aggregate_ip2(int K)
{
    IloEnv env;
    IloModel model(env);
    
    IloNumVar cplext(env, 0, 1, ILOFLOAT);
    
    vector<IloNumVar> cplexx(N);
    for (int i=0; i<N; ++i)
        cplexx[i] = IloNumVar(env, 0, 1, ILOBOOL);
    
    vector<vector<IloNumVar> > cplexmu(N);
    for (int i=0; i<N; ++i)
    {
        cplexmu[i].resize(N);
        for (int l=0; l<N; ++l)
            cplexmu[i][l] = IloNumVar(env, 0, 1, ILOFLOAT);
    }
    
    vector<IloNumVar> cplexts(N);
    for (int i=0; i<N; ++i)
        cplexts[i] = IloNumVar(env, 0, 1, ILOFLOAT);
    
    vector<vector<double> > d(N);
    for (int i=0; i<N; ++i)
    {
        d[i].resize(N,99999);
        for (int l=0; l<N; ++l)
            for (int s=0; s<n; ++s)
                d[i][l] = min(c[l][s]/c[i][s],d[i][l]);
    }
    
    for (int i=0; i<N; ++i)
    {
        IloExpr con(env);
        for (int l=0; l<N; ++l)
            con += d[i][l]*cplexmu[i][l];
        model.add(cplexts[i] <= con);
    }
    
    for (int i=0; i<N; ++i)
        for (int l=0; l<N; ++l)
            model.add(cplexmu[i][l] <= cplexx[l]);
                
    {
        IloExpr con(env);
        for (int i=0; i<N; ++i)
            con += cplexx[i];
        model.add(con == K);
    }
                
    for (int i=0; i<N; ++i)
        model.add(cplext <= cplexts[i]);
        
    for (int i=0; i<N; ++i)
    {
        IloExpr con(env);
        for (int l=0; l<N; ++l)
            con += cplexmu[i][l];
        model.add(con == 1);
    }
        
    IloExpr obj(env);
    for (int i=0; i<N; ++i)
        obj += cplexts[i];
    model.add(IloMaximize(env, cplext + 0.001*obj));
            
    IloCplex cplex(model);
    
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 1);
    if (timelimit > 0)
        cplex.setParam(IloCplex::TiLim, timelimit);
        
    bool result = cplex.solve();			
    
    double val=cplex.getValue(cplext);
    
    vector<vector<double> > scen;
    for (int i=0; i<N; ++i)
        if (cplex.getValue(cplexx[i]) > 0.5)
            scen.push_back(c[i]);
    
    vector<vector<double> > scen2(N);
    for (int i=0; i<N; ++i)
        scen2[i].resize(n);
    for (int i=0; i<N; ++i)
        for (int l=0; l<N; ++l)
            if (cplex.getValue(cplexmu[i][l]) > 0.5)
                for (int s=0; s<n; ++s)
                    scen2[l][s] = max(scen2[l][s],c[i][s]);
    
    env.end();
    
    return make_pair(scen,scen2);    
}
