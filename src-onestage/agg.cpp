#include "agg.h"
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "ilcplex/ilocplex.h"

ILOSTLBEGIN


using namespace std;

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
        }
        c.push_back(scen);
	}
	
	in.close();
}

void Agg::generate(int _n, int _N)
{
    n = _n;
    N = _N;

	c.resize(N);
    for (int i=0; i<N; ++i)
	{
		c[i].resize(n);
		for (int j=0; j<n; ++j)
            c[i][j] = rand()%100+1;
	}
}

void Agg::generate_outlier(int _n, int _N)
{
    n = _n;
    N = _N;
    
    c.resize(N);
    for (int i=0; i<N; ++i)
	{
		c[i].resize(n);
		for (int j=0; j<n; ++j)
            c[i][j] = rand()%100+1;
        if (rand()%100 < 5)
            for (int j=0; j<n; ++j)
                c[i][j] *= 2;
	}
}

void Agg::generate_2norm(int _n, int _N)
{
    n = _n;
    N = _N;

	c.resize(N);
    for (int i=0; i<N; ++i)
	{
        double sum = 0;
		c[i].resize(n);
		for (int j=0; j<n; ++j)
        {
            c[i][j] = rand()%100+1;
            sum += c[i][j]*c[i][j];
        }
        double r = 0.9 + 0.2*(rand()%101)/100.0;
        for (int j=0; j<n; ++j)
            c[i][j] = r*10000*(c[i][j]/sum);
	}
}
void Agg::generate_gamma(int _n, int _N)
{
    n = _n;
    N = _N;
    
    vector<double> clow(n);
    vector<double> d(n);
    vector<int> ind(n);
    
    for (int j=0; j<n; ++j)
    {
        clow[j] = rand()%100+1;
        d[j] = rand()%100+1;
        ind[j] = j;
    }
    
    
	c.resize(N);
    for (int i=0; i<N; ++i)
	{ 
        random_shuffle(ind.begin(), ind.end());
        c[i] = clow;
        for (int j=0; j<3; ++j)
            c[i][ind[j]] += d[ind[j]];
    }  
}
        

double Agg::evaluation(vector<double> x, vector<vector<double> > scen)
{
    int N = scen.size();
	vector<double> sum(N,0);
	for (int i=0; i<N; ++i)
		for (int j=0; j<n; ++j)
			sum[i] += scen[i][j] * x[j];

	double obj = 0;
	for (int i=0; i<N; ++i)
		if (sum[i]>=obj)
			obj = sum[i];

	return obj;
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
	
    IloNumVar cplext(env,0,IloInfinity,ILOFLOAT);
    
    for (int i=0; i<N; ++i)
	{ 
		IloExpr con(env);
		for (int j=0; j<n; ++j)
            con += scen[i][j] * cplexx[j];
		
		model.add(con<=cplext);
	}
    
    {
        IloExpr con(env);
        for (int j=0; j<n; ++j)
            con += cplexx[j];
        model.add(con == p);
    }
    
    model.add(IloMinimize(env,cplext));

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


vector<vector<double> > Agg::get_c()
{
    return c;
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

double Agg::distance(vector<double> vec1, vector<double> vec2)
{
    double sqrsum = 0;

    for (int j=0; j<n; ++j)
        sqrsum += (vec1[j] - vec2[j]) * (vec1[j] - vec2[j]);

    return sqrt(sqrsum);
}

void Agg::set_timelimit(double tl)
{
    timelimit = tl;
}


pair<double,vector<vector<double> > > Agg::aggregate_restart(int K, int R)
{
    vector<vector<double> > bestscen(K);
    for (int k=0; k<K; ++k)
        bestscen[k].resize(n,0);
    double bestt = 1000;
        
        
    for (int rep=0; rep<R; ++rep)
    {
        vector<vector<double> > lam(K);
        vector<int> available(N);
        for (int i=0; i<N; ++i)
            available[i] = i;
        random_shuffle(available.begin(), available.end());
        for (int k=0; k<K; ++k)
        {
            lam[k].resize(N,0);
            lam[k][available[k]] = 1;
        }
        
        double val= 1000;
        
        vector<vector<double> > scen(K);
        for (int k=0; k<K; ++k)
        {
            scen[k].resize(n,0);
            for (int s=0; s<n; ++s)
                for (int i=0; i<N; ++i)
                    scen[k][s] += c[i][s]*lam[k][i];
        }
        
        
        double nval = val;
        
        int itnumber = 0;
        do
        {
            ++itnumber;
            val = nval;
            
            vector<vector<double> > mu(K);
            for (int k=0; k<K; ++k)
                mu[k].resize(N,0);
                
            {
                IloEnv env;
                IloModel model(env);
        
                vector<IloNumVar> cplexts(N);
                for (int i=0; i<N; ++i)
                    cplexts[i] = IloNumVar(env,0,1,ILOFLOAT);
        
                vector<vector<IloNumVar> > cplexmu(K);
                for (int k=0; k<K; ++k)
                {
                    cplexmu[k].resize(N);
                    for (int i=0; i<N; ++i)
                        cplexmu[k][i] = IloNumVar(env, 0, 1, ILOFLOAT);
                }
        
        
                for (int i=0; i<N; ++i)
                    for (int s=0; s<n; ++s)
                    {
                        IloExpr con(env);
                        for (int k=0; k<K; ++k)
                            for (int l=0; l<N; ++l)
                                con += lam[k][l]*c[l][s]*cplexmu[k][i];
                            
                        model.add(c[i][s] * cplexts[i] <= con);
                    }
                    
                for (int i=0; i<N; ++i)
                {
                    IloExpr con(env);
                    for (int k=0; k<K; ++k)
                        con += cplexmu[k][i];
                    model.add(con == 1);
                }
                    
                IloExpr obj(env);
                for (int i=0; i<N; ++i)
                    obj += cplexts[i];
                model.add(IloMaximize(env, obj));
                
                IloCplex cplex(model);
                
                cplex.setOut(env.getNullStream());
                cplex.setParam(IloCplex::Threads, 1);
                
                bool result = cplex.solve();			
                
                for (int k=0; k<K; ++k)
                    for (int i=0; i<N; ++i)
                        mu[k][i] = cplex.getValue(cplexmu[k][i]);
                
                env.end();
            }
            
            {
                IloEnv env;
                IloModel model(env);
                
                IloNumVar cplext(env, 0, 1, ILOFLOAT);
                
                vector<vector<IloNumVar> > cplexlambda(K);
                for (int k=0; k<K; ++k)
                {
                    cplexlambda[k].resize(N);
                    for (int i=0; i<N; ++i)
                        cplexlambda[k][i] = IloNumVar(env, 0, 1, ILOFLOAT);
                }
                
                for (int i=0; i<N; ++i)
                    for (int s=0; s<n; ++s)
                    {
                        IloExpr con(env);
                        for (int k=0; k<K; ++k)
                            for (int l=0; l<N; ++l)
                                con += c[l][s]*cplexlambda[k][l]*mu[k][i];
                        model.add(c[i][s] * cplext <= con);
                    }
                    
                for (int k=0; k<K; ++k)
                {
                    IloExpr con(env);
                    for (int i=0; i<N; ++i)
                        con += cplexlambda[k][i];
                    model.add(con == 1);
                }
                
                model.add(IloMaximize(env, cplext));
                        
                IloCplex cplex(model);
                
                cplex.setOut(env.getNullStream());
                cplex.setParam(IloCplex::Threads, 1);
                
                bool result = cplex.solve();			
                
                nval= 1/cplex.getValue(cplext);
                
                for (int k=0; k<K; ++k)
                    for (int i=0; i<N; ++i)
                        lam[k][i] = cplex.getValue(cplexlambda[k][i]);

                for (int k=0; k<K; ++k)
                    for (int s=0; s<n; ++s)
                    {
                        scen[k][s] = 0;
                        for (int i=0; i<N; ++i)
                            scen[k][s] += c[i][s]*cplex.getValue(cplexlambda[k][i]);
                    }
                    
                    
                env.end();
            }
            
        }while(nval +0.00001< val && itnumber<20);
        
        if (nval < bestt)
        {
            bestt = nval;
            bestscen = scen;
        }    
    }
    
    
    return make_pair(bestt,bestscen);    
}



pair<double,vector<vector<double> > > Agg::aggregate_ip_lambda2(int K)
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
    
    for (int i=0; i<N; ++i)
        for (int s=0; s<n; ++s)
        {
            IloExpr con(env);
            for (int l=0; l<N; ++l)
                con += cplexmu[i][l]*c[l][s];
            model.add(c[i][s] * cplexts[i] <= con);
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
    
    env.end();
    
    return make_pair(1/val,scen);   
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
	
    IloNumVar cplext(env,0,IloInfinity,ILOFLOAT);
    
    for (int i=0; i<N; ++i)
	{ 
		IloExpr con(env);
		for (int j=0; j<n; ++j)
            con += scen[i][j] * cplexx[j];
		
		model.add(con<=cplext);
	}
    
    for (int j=0; j<n; ++j)
    {
        IloExpr con(env);
        for (int l=0; l<edges.size(); ++l)
        {
            if (edges[l].first == j)
                con += cplexx[edges[l].second];
            if (edges[l].second == j)
                con += cplexx[edges[l].first];
        }
        con += cplexx[j];
        model.add(con >= 1);
    }
    
    model.add(IloMinimize(env,cplext));

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




pair<double,vector<vector<double> > > Agg::aggregate_ip2_warmstart(int K)
{
    vector<vector<double> > kscen = aggregate_kmeans(K,10);
    vector<vector<int> > mu(K);
    for (int k=0; k<K; ++k)
        mu[k].resize(N);
    for (int i=0; i<N; ++i)
    {
        int mink = 0;
        double mind = distance(kscen[0],c[i]);
        for (int k=1; k<K; ++k)
        {
            double dist = distance(kscen[k],c[i]);
            if (dist < mind)
            {
                mind = dist;
                mink = k;
            }
        }
        mu[mink][i] = 1;
    }
    
    
    IloEnv env;
    IloModel model(env);
    
    IloNumVar cplext(env, 0, 1, ILOFLOAT);
    
    vector<vector<IloNumVar> > cplexlambda(K);
    for (int k=0; k<K; ++k)
    {
        cplexlambda[k].resize(N);
        for (int i=0; i<N; ++i)
            cplexlambda[k][i] = IloNumVar(env, 0, 1, ILOFLOAT);
    }
    
    vector<vector<IloNumVar> > cplexmu(K);
    for (int k=0; k<K; ++k)
    {
        cplexmu[k].resize(N);
        for (int i=0; i<N; ++i)
            cplexmu[k][i] = IloNumVar(env, 0, 1, ILOBOOL);
    }
    
    vector<IloNumVar> cplexts(N);
    for (int i=0; i<N; ++i)
        cplexts[i] = IloNumVar(env, 0, 1, ILOFLOAT);
    
    for (int i=0; i<N; ++i)
        for (int s=0; s<n; ++s)
            for (int k=0; k<K; ++k)
            {
                IloExpr con(env);
                for (int l=0; l<N; ++l)
                    con += c[l][s]*cplexlambda[k][l];
                model.add(c[i][s] * cplexts[i] <= con + 100*(1-cplexmu[k][i]));
            }
        
    for (int i=0; i<N; ++i)
        model.add(cplext <= cplexts[i]);
        
    for (int k=0; k<K; ++k)
    {
        IloExpr con(env);
        for (int i=0; i<N; ++i)
            con += cplexlambda[k][i];
        model.add(con == 1);
    }
    
    for (int i=0; i<N; ++i)
    {
        IloExpr con(env);
        for (int k=0; k<K; ++k)
            con += cplexmu[k][i];
        model.add(con == 1);
    }
        
    IloExpr obj(env);
    for (int i=0; i<N; ++i)
        obj += cplexts[i];
    model.add(IloMaximize(env, cplext + 0.001*obj));
            
    IloCplex cplex(model);
    
    
    IloNumVarArray startVar(env);
    IloNumArray startVal(env);
    for (int k=0; k<K; ++k)
        for (int i=0; i<N; ++i)
        {
            startVar.add(cplexmu[k][i]);
            startVal.add(mu[k][i]);
        }
            
    cplex.addMIPStart(startVar, startVal);
    startVal.end();
    startVar.end();
    
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 1);
    if (timelimit > 0)
        cplex.setParam(IloCplex::TiLim, timelimit);
        
    bool result = cplex.solve();			
    
    double val=cplex.getValue(cplext);
    
    vector<vector<double> > scen(K);
    for (int k=0; k<K; ++k)
    {
        scen[k].resize(n,0);
        for (int s=0; s<n; ++s)
            for (int i=0; i<N; ++i)
                scen[k][s] += c[i][s]*cplex.getValue(cplexlambda[k][i]);
    }
        
    env.end();
    
    return make_pair(1/val,scen);
}
