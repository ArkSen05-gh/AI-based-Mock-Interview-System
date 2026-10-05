import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Brain,
  Code,
  Briefcase,
  Wrench,
  Zap,
  TrendingUp,
  Database,
  LogOut,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

/* ---------------- DOMAIN DATA ---------------- */

const domains = [
  {
    id: "computer-science",
    name: "Computer Science",
    icon: Code,
    description: "Data Structures, Algorithms, System Design",
  },
  {
    id: "information-technology",
    name: "Information Technology",
    icon: Database,
    description: "Networks, Security, Cloud Computing",
  },
  {
    id: "mechanical",
    name: "Mechanical Engineering",
    icon: Wrench,
    description: "Thermodynamics, Mechanics, Design",
  },
  {
    id: "electrical",
    name: "Electrical Engineering",
    icon: Zap,
    description: "Circuits, Power Systems, Electronics",
  },
  {
    id: "business-analytics",
    name: "Business Analytics",
    icon: TrendingUp,
    description: "Data Analysis, Statistics, Insights",
  },
  {
    id: "general-business",
    name: "General Business",
    icon: Briefcase,
    description: "Management, Operations, Strategy",
  },
];

/* ---------------- COMPANY DATA ---------------- */

const companies = [
  { id: "wipro", name: "Wipro" },
  { id: "tcs", name: "TCS" },
  { id: "deloitte", name: "Deloitte" },
  { id: "pwc", name: "PwC" },
  { id: "infosys", name: "Infosys" },
  { id: "ibm", name: "IBM" },
  { id: "cognizant", name: "Cognizant" },
  { id: "accenture", name: "Accenture" },
  { id: "hcltech", name: "HCLTech" },
  { id: "ltimindtree", name: "LTIMindtree" },
];

const Dashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  /* ---------------- AUTH CHECK ---------------- */

  useEffect(() => {
    const checkUser = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) {
        navigate("/auth");
        return;
      }

      setUser(session.user);
      setLoading(false);
    };

    checkUser();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) navigate("/auth");
      else setUser(session.user);
    });

    return () => subscription.unsubscribe();
  }, [navigate]);

  /* ---------------- HANDLERS ---------------- */

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    toast({
      title: "Signed Out",
      description: "You've been successfully signed out.",
    });
    navigate("/");
  };

  const handleDomainSelect = (domainId: string) => {
    navigate(`/interview?domain=${domainId}`);
  };

  const handleCompanyDomainSelect = (
    companyId: string,
    domainId: string
  ) => {
    navigate(`/interview?company=${companyId}&domain=${domainId}`);
  };

  /* ---------------- LOADING ---------------- */

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    );
  }

  /* ---------------- UI ---------------- */

  return (
    <div className="min-h-screen bg-background">
      {/* HEADER */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold">InterviewIQ</h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">
              {user?.user_metadata?.full_name || user?.email}
            </span>
            <Button variant="outline" size="sm" onClick={handleSignOut}>
              <LogOut className="h-4 w-4 mr-2" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main className="container mx-auto px-4 py-12 space-y-16">
        {/* DOMAIN SECTION */}
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-4xl font-bold">Choose Your Domain</h2>
            <p className="text-muted-foreground mt-2">
              Practice Interviews by Domain
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {domains.map((domain) => {
              const Icon = domain.icon;
              return (
                <Card
                  key={domain.id}
                  className="cursor-pointer transition-all hover:shadow-strong hover:scale-105 gradient-card"
                  onClick={() => handleDomainSelect(domain.id)}
                >
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Icon className="h-6 w-6 text-primary" />
                      </div>
                      <CardTitle className="text-xl">
                        {domain.name}
                      </CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base">
                      {domain.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* COMPANY + DOMAIN SECTION */}
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold">
              Company Wise Practice Interview
            </h2>
            <p className="text-muted-foreground mt-2">
              Choose a Company and Domain
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {companies.map((company) => (
              <Card key={company.id} className="shadow-soft">
                <CardHeader>
                  <CardTitle className="text-xl text-center">
                    {company.name}
                  </CardTitle>
                  <CardDescription className="text-center">
                    Select a domain
                  </CardDescription>
                </CardHeader>

                <CardContent className="grid grid-cols-2 gap-3">
                  {domains.map((domain) => (
                    <Button
                      key={domain.id}
                      variant="outline"
                      className="text-sm"
                      onClick={() =>
                        handleCompanyDomainSelect(
                          company.id,
                          domain.id
                        )
                      }
                    >
                      {domain.name}
                    </Button>
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* HOW IT WORKS */}
        <Card>
          <CardHeader>
            <CardTitle>How It Works</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {[
              "Select a company and domain",
              "Answer AI-generated interview questions",
              "Get detailed feedback and improvement plan",
            ].map((step, i) => (
              <div key={i} className="flex gap-3 items-center">
                <div className="h-8 w-8 rounded-full bg-primary text-white flex items-center justify-center">
                  {i + 1}
                </div>
                <p className="text-sm text-muted-foreground">{step}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Dashboard;