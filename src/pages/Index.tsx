import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";
import { Brain, Video, LineChart, Target, Sparkles, CheckCircle2 } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Video,
      title: "AI-Powered Analysis",
      description: "Real-time evaluation of facial expressions, eye movement, and speech patterns",
    },
    {
      icon: LineChart,
      title: "Multi-Dimensional Scoring",
      description: "Comprehensive assessment of confidence, communication, and technical accuracy",
    },
    {
      icon: Target,
      title: "Personalized Feedback",
      description: "Detailed insights and actionable recommendations for improvement",
    },
    {
      icon: Sparkles,
      title: "Progress Tracking",
      description: "Monitor your improvement over time with detailed analytics and benchmarking",
    },
  ];

  const benefits = [
    "Practice with domain-specific questions",
    "Get instant AI-powered feedback",
    "Track your progress over time",
    "Compare performance with peers",
    "Build confidence for real interviews",
    "Access learning resources tailored to your weaknesses",
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="gradient-hero py-20 px-4">
        <div className="container mx-auto">
          <div className="max-w-4xl mx-auto text-center text-white space-y-8 animate-fade-in">
            <div className="flex items-center justify-center gap-3 mb-6">
              <Brain className="h-16 w-16 animate-float" />
              <h1 className="text-5xl md:text-6xl font-bold">InterviewIQ</h1>
            </div>
            <p className="text-xl md:text-2xl opacity-90">
              Master Your Interviews with AI-Powered Practice
            </p>
            <p className="text-lg opacity-80 max-w-2xl mx-auto">
              Prepare for corporate, technical, and business interviews with our advanced AI system that analyzes your performance in real-time
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
              <Button
                size="lg"
                variant="secondary"
                className="text-lg px-8 py-6 shadow-strong"
                onClick={() => navigate("/auth")}
              >
                Get Started Free
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="text-lg px-8 py-6 bg-white/10 border-white/20 text-white hover:bg-white/20"
                onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              >
                Learn More
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4">Cutting-Edge Features</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Our AI analyzes multiple dimensions of your interview performance to provide comprehensive feedback
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <Card key={feature.title} className="shadow-soft gradient-card hover:shadow-strong transition-all">
                  <CardHeader>
                    <div className="p-3 rounded-lg bg-primary/10 w-fit mb-2">
                      <Icon className="h-8 w-8 text-primary" />
                    </div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 px-4 bg-secondary/30">
        <div className="container mx-auto">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold mb-4">Why Choose InterviewIQ?</h2>
              <p className="text-lg text-muted-foreground">
                Everything you need to ace your next interview
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {benefits.map((benefit, index) => (
                <div key={index} className="flex items-start gap-3 p-4 rounded-lg bg-card shadow-soft">
                  <CheckCircle2 className="h-6 w-6 text-accent shrink-0 mt-0.5" />
                  <p className="text-base">{benefit}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <Card className="max-w-3xl mx-auto shadow-strong gradient-card text-center">
            <CardHeader className="space-y-4">
              <CardTitle className="text-3xl md:text-4xl">
                Ready to Transform Your Interview Skills?
              </CardTitle>
              <CardDescription className="text-lg">
                Join thousands of candidates who have improved their interview performance with InterviewIQ
              </CardDescription>
            </CardHeader>
            <CardContent className="pb-8">
              <Button
                size="lg"
                className="text-lg px-8 py-6"
                onClick={() => navigate("/auth")}
              >
                Start Practicing Now
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8 px-4">
        <div className="container mx-auto text-center text-muted-foreground">
          <p>© 2025 InterviewIQ. Powered by AI and NLP technology.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
