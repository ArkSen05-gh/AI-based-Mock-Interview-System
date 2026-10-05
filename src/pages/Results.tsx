import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Brain,
  TrendingUp,
  MessageSquare,
  Target,
  Lightbulb,
  Home,
  BookOpen,
  CheckCircle,
  XCircle,
  Percent,
  ListChecks,
  Activity, // 🔥 ADDED
} from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";

/* ---------------- TYPES ---------------- */

interface EvaluationResult {
  confidence: number;
  clarity: number;
  accuracy: number;
  nervousness_index: number | null;
  overall: number;
  feedback: string;
  detail_per_question: {
    question: string;
    transcript?: string;
    nervousness_index?: number;
    mistakes?: string;
  }[];
  domain?: string;
}



const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [data, setData] = useState<EvaluationResult | null>(null);
  const [isSilentInterview, setIsSilentInterview] = useState(false);
  const [researchLoading, setResearchLoading] = useState(true);
  const [researchError, setResearchError] = useState<string | null>(null);

  /* ---------------- STRESS STATES ---------------- */
  const [stressScore, setStressScore] = useState(0);
  const [stressTimeline, setStressTimeline] = useState<{time:string; stress:number}[]>([]);

  /* ---------------- RESEARCH STATES ---------------- */
  const [research, setResearch] = useState<any>(null);
  const [researchStressTimeline, setResearchStressTimeline] = useState<any[]>([]);
  const [researchConfTimeline, setResearchConfTimeline] = useState<any[]>([]);
  const [fsmTimeline, setFsmTimeline] = useState<any[]>([]);

  /* ---------------- PLOT STATES ---------------- */
  const [plots, setPlots] = useState<{ name: string; url: string }[]>([]);
  const [plotsLoading, setPlotsLoading] = useState(false);
  const [plotsError, setPlotsError] = useState<string | null>(null);
  const [participants, setParticipants] = useState<{ id: string; file_count: number }[]>([]);
  const [selectedParticipant, setSelectedParticipant] = useState<string>("");

  /* ---------------- INTERVIEW SESSION RESULTS (REST API) ---------------- */
  const [interviewSessionId, setInterviewSessionId] = useState<string>("");
  const [interviewResults, setInterviewResults] = useState<any>(null);
  const [interviewLoading, setInterviewLoading] = useState(false);
  const [interviewError, setInterviewError] = useState<string | null>(null);

  /* ---------------- AUTH CHECK ---------------- */
  useEffect(() => {
    const checkAuth = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) navigate("/auth");
    };
    checkAuth();
  }, [navigate]);

  /* Helper: apply research data to state */
  const applyResearchData = (researchData: any) => {
    setResearch(researchData);

    // Build stress + confidence dual timeline
    const stressSeries = researchData.stress_series || [];
    const confSeries = researchData.confidence_series || [];
    const timeSecs = researchData.time_seconds || [];
    const maxLen = Math.max(stressSeries.length, confSeries.length);
    const dualTimeline = [];
    for (let i = 0; i < maxLen; i++) {
      dualTimeline.push({
        time: timeSecs[i] ? `${Math.round(timeSecs[i])}s` : `${i}`,
        stress: Math.round((stressSeries[i] || 0) * 100),
        confidence: Math.round((confSeries[i] || 0) * 100),
      });
    }
    setResearchStressTimeline(dualTimeline);

    // FSM state timeline
    const fsmSeries = researchData.fsm_state_series || [];
    const fsmTimelineData = fsmSeries.map((state: string, i: number) => ({
      time: `${i}`,
      state,
      stateId: ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"].indexOf(state),
    }));
    setFsmTimeline(fsmTimelineData);
  };

  /* ---------------- LOAD RESULT ---------------- */
  useEffect(() => {
    const stored = localStorage.getItem("last_evaluation");
    if (!stored) {
      setResearchLoading(false);
      return;
    }

    try {
      const parsed: EvaluationResult = JSON.parse(stored);

      const hasAnyAnswer = parsed.detail_per_question.some(
        (q) => q.transcript && q.transcript.trim().length > 0
      );

      if (!hasAnyAnswer) {
        setIsSilentInterview(true);
        setData({
          ...parsed,
          confidence: 0,
          clarity: 0,
          accuracy: 0,
          nervousness_index: null,
          overall: 0,
        });
      } else {
        setIsSilentInterview(false);
        setData(parsed);
      }

      if (parsed.nervousness_index !== null) {
        setStressScore(Math.round(parsed.nervousness_index));
      }

      const safeStress = (val: any, fallback: number) => {
        if (val === null || val === undefined || isNaN(Number(val))) return fallback;
        return Math.round(Number(val));
      };
      
      const fallbackTimeline = parsed.detail_per_question.map((q, i) => ({
        time: `Q${i + 1}`,
        stress: q.nervousness_index != null 
          ? safeStress(q.nervousness_index, 0) 
          : safeStress(parsed.nervousness_index, 0)
      }));

      // Load real-time stress history from WebSocket session
      const storedHistory = localStorage.getItem("realtime_stress_history");
      if (storedHistory) {
        const parsedHistory = JSON.parse(storedHistory);
        if (Array.isArray(parsedHistory) && parsedHistory.length > 0) {
          setStressTimeline(parsedHistory.map(p => ({
            time: p.time || "Unknown",
            stress: safeStress(p.stress, 0)
          })));
        } else {
          setStressTimeline(fallbackTimeline);
        }
      } else {
        setStressTimeline(fallbackTimeline);
      }

      // Load research data if available (from localStorage fallback)
      const researchData = (parsed as any).research;
      if (researchData && !researchData.error) {
        applyResearchData(researchData);
      }

      // Fetch the full research summary from the backend API
      const sessionId = localStorage.getItem("interview_session_id");
      if (sessionId) {
        setResearchLoading(true);
        fetch(`http://localhost:8000/api/sessions/${sessionId}/research-summary`)
          .then((res) => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
          })
          .then((apiResearch) => {
            if (apiResearch && !apiResearch.error) {
              applyResearchData(apiResearch);
              setResearchError(null);
            }
          })
          .catch((err) => {
            console.warn("Could not fetch research summary from API:", err);
            setResearchError("Could not load research data from server.");
          })
          .finally(() => setResearchLoading(false));
      } else {
        setResearchLoading(false);
      }
    } catch (err) {
      console.error("Failed to parse evaluation:", err);
      setResearchLoading(false);
    }
  }, []);

  /* ---------------- LOAD PARTICIPANTS ---------------- */
  useEffect(() => {
    fetch("http://localhost:8000/participants")
      .then((res) => res.json())
      .then((data) => {
        if (data && Array.isArray(data.participants)) {
          setParticipants(data.participants);
          if (data.participants.length > 0) {
            setSelectedParticipant("_all");
          }
        }
      })
      .catch(() => {
        setParticipants([]);
      });
  }, []);

  /* ---------------- INTERVIEW SESSION RESULTS (REST API) ---------------- */
  useEffect(() => {
    const navState = (location.state as { sessionId?: string } | null) ?? null;
    const sessionId =
      navState?.sessionId ||
      (location as any).query?.sessionId ||
      "";
    if (!sessionId) {
      setInterviewLoading(false);
      return;
    }
    setInterviewSessionId(sessionId);
    setInterviewLoading(true);
    setInterviewError(null);

    let cancelled = false;
    const attempt = (tries: number) => {
      fetch(`http://localhost:8000/interview/results/${encodeURIComponent(sessionId)}`)
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((payload) => {
          if (cancelled) return;
          const status = payload?.status;
          if (status === "completed") {
            setInterviewResults(payload);
            setInterviewLoading(false);
            return;
          }
          if (tries < 20) {
            setTimeout(() => attempt(tries + 1), 1500);
          } else {
            setInterviewError("Interview results are still being processed.");
            setInterviewLoading(false);
          }
        })
        .catch((err) => {
          if (cancelled) return;
          if (tries < 5) {
            setTimeout(() => attempt(tries + 1), 1500);
          } else {
            setInterviewError(
              `Could not load interview results: ${err?.message || err}`
            );
            setInterviewLoading(false);
          }
        });
    };
    attempt(0);
    return () => {
      cancelled = true;
    };
  }, [location.state]);

  /* ---------------- GENERATE / LOAD PLOTS ---------------- */
  const loadParticipantPlots = (pid: string) => {
    if (!pid) return;
    setPlotsLoading(true);
    setPlotsError(null);
    setPlots([]);

    // Try to load existing plots first
    fetch(`http://localhost:8000/analysis/${encodeURIComponent(pid)}/plots`)
      .then((res) => res.json())
      .then((data) => {
        if (data && Array.isArray(data.plots) && data.plots.length > 0) {
          setPlots(data.plots);
          setPlotsLoading(false);
        } else {
          // No plots yet — generate them
          return fetch(`http://localhost:8000/analysis/${encodeURIComponent(pid)}/generate_plots`, {
            method: "POST",
          }).then((res) => res.json());
        }
      })
      .then((gen: any) => {
        if (gen && Array.isArray(gen.plots)) {
          setPlots(gen.plots);
        }
      })
      .catch((err) => {
        console.error("Could not load plots:", err);
        setPlotsError("Could not load participant plots.");
      })
      .finally(() => setPlotsLoading(false));
  };

  /* ---------------- LOAD ALL-PARTICIPANT PLOTS ---------------- */
  const loadAllParticipantPlots = () => {
    setPlotsLoading(true);
    setPlotsError(null);
    setPlots([]);

    // Try to load existing aggregate plots first
    fetch("http://localhost:8000/analysis/_all/plots")
      .then((res) => res.json())
      .then((data) => {
        if (data && Array.isArray(data.plots) && data.plots.length > 0) {
          setPlots(data.plots);
          setPlotsLoading(false);
        } else {
          // No plots yet — generate them
          return fetch("http://localhost:8000/analysis/_all/generate_plots", {
            method: "POST",
          }).then((res) => res.json());
        }
      })
      .then((gen: any) => {
        if (gen && Array.isArray(gen.plots)) {
          setPlots(gen.plots);
        }
      })
      .catch((err) => {
        console.error("Could not load all-participant plots:", err);
        setPlotsError("Could not load all-participant plots.");
      })
      .finally(() => setPlotsLoading(false));
  };

  /* --------- GENERATE PLOTS ON PARTICIPANT SELECT --------- */
  useEffect(() => {
    if (selectedParticipant === "_all") {
      loadAllParticipantPlots();
    } else if (selectedParticipant) {
      loadParticipantPlots(selectedParticipant);
    }
  }, [selectedParticipant]);

  if (!data) return null;

  /* ---------------- DERIVED METRICS ---------------- */
  const totalQuestions = data.detail_per_question.length;

  const answeredCount = data.detail_per_question.filter(
    (q) => q.transcript && q.transcript.trim().length > 0
  ).length;

  const skippedCount = totalQuestions - answeredCount;

  const completionRate =
    totalQuestions === 0
      ? 0
      : Math.round((answeredCount / totalQuestions) * 100);

  /* ---------------- CATEGORY SCORES ---------------- */
  const scores = {
    confidence: isSilentInterview ? 0 : data.confidence,
    communication: isSilentInterview ? 0 : data.clarity,
    technicalAccuracy: isSilentInterview ? 0 : data.accuracy,
    emotionalStability:
      isSilentInterview || data.nervousness_index === null
        ? 0
        : 100 - data.nervousness_index,
    overall: isSilentInterview ? 0 : data.overall,
  };

  /* ---------------- FEEDBACK CATEGORIES ---------------- */
  const feedback = [
    {
      category: "Confidence",
      icon: TrendingUp,
      score: scores.confidence,
      comment: "Based on fluency, response length, and hesitation patterns.",
    },
    {
      category: "Communication",
      icon: MessageSquare,
      score: scores.communication,
      comment: "Measures clarity and filler word usage.",
    },
    {
      category: "Technical Accuracy",
      icon: Target,
      score: scores.technicalAccuracy,
      comment: "Relevance of concepts and technical correctness.",
    },
    {
      category: "Emotional Stability",
      icon: Brain,
      score: scores.emotionalStability,
      comment: "Derived from hesitation and response confidence.",
    },
    {
      category: "Answered Questions",
      icon: CheckCircle,
      score: (answeredCount / totalQuestions) * 100,
      comment: `${answeredCount} questions answered`,
    },
    {
      category: "Skipped Questions",
      icon: XCircle,
      score: (skippedCount / totalQuestions) * 100,
      comment: `${skippedCount} questions skipped`,
    },
    {
      category: "Completion Rate",
      icon: Percent,
      score: completionRate,
      comment: `Interview completion: ${completionRate}%`,
    },
  ];



  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* HEADER */}
      <header className="border-b bg-card/80 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary animate-pulse" />
            <h1 className="text-2xl font-bold text-foreground tracking-tight">InterviewIQ</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/dashboard")} className="border-border hover:bg-secondary">
            <Home className="h-4 w-4 mr-2" />
            Dashboard
          </Button>
        </div>
      </header>

      {/* MAIN */}
      <main className="container mx-auto px-4 py-12 flex-1 flex flex-col justify-start">
        <div className="max-w-4xl mx-auto w-full space-y-8 animate-fade-in">
          <div className="text-center space-y-2">
            <h2 className="text-4xl font-bold tracking-tight text-foreground">Interview Results</h2>
            <p className="text-muted-foreground mt-2">
              Answered {answeredCount} / {totalQuestions} • Skipped{" "}
              {skippedCount}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* OVERALL SCORE CARD */}
            <Card className="shadow-soft border-border bg-card gradient-card hover:shadow-strong transition-all flex flex-col justify-between">
              <CardHeader className="text-center pb-2">
                <CardTitle className="text-2xl font-bold text-foreground">Overall Performance Score</CardTitle>
                <CardDescription className="text-sm text-muted-foreground">
                  Aggregated based on communication, accuracy, and confidence
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center space-y-4 pt-4">
                <div className="text-7xl font-extrabold text-primary">
                  {scores.overall.toFixed(1)}%
                </div>
                <Progress value={scores.overall} className="h-3 bg-secondary" />
              </CardContent>
            </Card>

            {/* Silent interview note */}
            {isSilentInterview && (
              <Card className="col-span-full shadow-soft border-amber-300/40 bg-amber-500/5">
                <CardContent className="py-4 text-center">
                  <p className="text-sm text-amber-700 dark:text-amber-400 font-medium">
                    All questions were skipped; no transcript answers to evaluate. Research metrics below are still available from video/audio analysis.
                  </p>
                </CardContent>
              </Card>
            )}

{/* QUESTIONS ANSWERED CARD */}
<Card className="shadow-soft border-border bg-card gradient-card hover:shadow-strong transition-all">
  <CardHeader className="text-center pb-2">
    <CardTitle className="text-2xl font-bold text-foreground">Questions Answered</CardTitle>
  </CardHeader>
  <CardContent className="text-center space-y-2 pt-4">
    <div className="text-5xl font-extrabold text-primary">{answeredCount}</div>
    <Progress value={(answeredCount/totalQuestions)*100} className="h-2 bg-secondary" />
  </CardContent>
</Card>

{/* QUESTIONS UNANSWERED CARD */}
<Card className="shadow-soft border-border bg-card gradient-card hover:shadow-strong transition-all">
  <CardHeader className="text-center pb-2">
    <CardTitle className="text-2xl font-bold text-foreground">Questions Unanswered</CardTitle>
  </CardHeader>
  <CardContent className="text-center space-y-2 pt-4">
    <div className="text-5xl font-extrabold text-primary">{skippedCount}</div>
    <Progress value={(skippedCount/totalQuestions)*100} className="h-2 bg-secondary" />
  </CardContent>
</Card>

{/* TECHNICAL ACCURACY CARD */}
<Card className="shadow-soft border-border bg-card gradient-card hover:shadow-strong transition-all">
  <CardHeader className="text-center pb-2">
    <CardTitle className="text-2xl font-bold text-foreground">Technical Accuracy</CardTitle>
  </CardHeader>
  <CardContent className="text-center space-y-2 pt-4">
    <div className="text-5xl font-extrabold text-primary">{scores.technicalAccuracy}%</div>
    <Progress value={scores.technicalAccuracy} className="h-2 bg-secondary" />
  </CardContent>
</Card>
            {/* REAL-TIME STRESS DETECTED SPIKE CARD */}
            <Card className="shadow-soft border-border bg-card gradient-card hover:shadow-strong transition-all">
              <CardHeader className="flex-row items-center gap-3 pb-2">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Activity className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <CardTitle className="text-2xl font-bold text-foreground">Real-time Stress Spike Analysis</CardTitle>
                  <CardDescription className="text-sm text-muted-foreground">
                    Live telemetry detected via MediaPipe & Python backend
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 pt-4">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground">Stress Level Indicator</span>
                  <span
                    className={`text-2xl font-bold ${
                      stressScore > 70
                        ? "text-destructive"
                        : stressScore > 40
                        ? "text-yellow-500"
                        : "text-green-500"
                    }`}
                  >
                    {stressScore}/100 ({stressScore > 70 ? "Anxious" : stressScore > 40 ? "Stressed" : "Calm"})
                  </span>
                </div>
                <Progress value={stressScore} className="h-3 bg-secondary" />
              </CardContent>
            </Card>
          </div>

          {/* REALTIME STRESS GRAPH SPIKES */}
          <Card className="shadow-soft border-border bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-xl font-bold text-foreground flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Real-time Stress Spike Timeline
              </CardTitle>
              <CardDescription className="text-sm text-muted-foreground">
                Continuous nervousness and vocal tension fluctuation over the course of the session
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stressTimeline}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 10, fill: "#6b7280" }}
                      axisLine={false}
                      tickLine={false}
                      minTickGap={45}
                    />
                    <YAxis 
                      domain={[0, 100]} 
                      tick={{ fontSize: 11, fill: "#6b7280" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: "#ffffff", 
                        borderColor: "#e5e7eb",
                        borderRadius: "12px", 
                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
                        color: "#111827"
                      }}
                      labelStyle={{ fontWeight: "bold" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="stress"
                      name="Stress Level"
                      stroke="#ef4444"
                      strokeWidth={3}
                      dot={{ r: 4, fill: "#ef4444", strokeWidth: 0 }}
                      activeDot={{ r: 6, fill: "#ef4444", strokeWidth: 0 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* ═══════════════════════════════════════════════════════════════
              PANEL A — Research Analysis: Stress & Confidence Timeline
              ═══════════════════════════════════════════════════════════════ */}
          {research && researchStressTimeline.length > 0 && (
            <Card className="shadow-soft border-border bg-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-xl font-bold text-foreground flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  Research Analysis — Stress &amp; Confidence Timeline
                </CardTitle>
                <CardDescription className="text-sm text-muted-foreground">
                  BiLSTM temporal stress proxy (red) and AU-CNN confidence estimator (blue) — Eqs. 3.1-3.3
                  {research.changepoints?.length > 0 && ` · ${research.changepoints.length} changepoint(s) detected (Δ)`}
                  {research.bilstm_spikes?.length > 0 && ` · ${research.bilstm_spikes.length} stress spike(s) flagged`}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={researchStressTimeline}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                      <XAxis
                        dataKey="time"
                        label={{ value: "Time (seconds)", position: "bottom", offset: -5, style: { fontSize: 11, fill: "#6b7280" } }}
                        tick={{ fontSize: 10, fill: "#6b7280" }}
                        axisLine={false}
                        tickLine={false}
                        minTickGap={45}
                      />
                      <YAxis
                        domain={[0, 100]}
                        label={{ value: "%", position: "insideTopLeft", offset: 10, style: { fontSize: 11, fill: "#6b7280" } }}
                        tick={{ fontSize: 11, fill: "#6b7280" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#fff", borderColor: "#e5e7eb", borderRadius: "12px", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)", color: "#111827" }}
                        labelStyle={{ fontWeight: "bold" }}
                      />
                      {/* Changepoint reference lines (Δ) */}
                      {research.changepoints?.map((cpIdx: number, i: number) => {
                        const timeLabel = researchStressTimeline[cpIdx]?.time || `${cpIdx}`;
                        return (
                          <ReferenceLine
                            key={`cp-${i}`}
                            x={timeLabel}
                            stroke="#f59e0b"
                            strokeDasharray="6 4"
                            strokeWidth={1.5}
                            label={{ value: "Δ", position: "top", fill: "#f59e0b", fontSize: 14, fontWeight: "bold" }}
                          />
                        );
                      })}
                      <Line type="monotone" dataKey="stress" name="Stress %" stroke="#ef4444" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
                      <Line type="monotone" dataKey="confidence" name="Confidence %" stroke="#3b82f6" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* Spike markers legend */}
                {research.bilstm_spikes?.length > 0 && (
                  <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-3 h-3 rounded-full bg-red-500" /> Stress Spike
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-4 border-t-2 border-dashed border-amber-500" /> Changepoint (Δ)
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              PANEL B — FSM Adaptive Questioning State Timeline
              ═══════════════════════════════════════════════════════════════ */}
          {research && fsmTimeline.length > 0 && (
            <Card className="shadow-soft border-border bg-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-xl font-bold text-foreground flex items-center gap-2">
                  <Activity className="h-5 w-5 text-violet-500" />
                  FSM State Over Time
                </CardTitle>
                <CardDescription className="text-sm text-muted-foreground">
                  Five-state FSM controller transitions (Baseline → Monitor → Adapt → Recover → Escalate) — Eq. 3.4
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fsmTimeline}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                      <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#6b7280" }} axisLine={false} tickLine={false} label={{ value: "Frame Index", position: "bottom", offset: -5, style: { fontSize: 11, fill: "#6b7280" } }} />
                      <YAxis
                        domain={[0, 4]}
                        ticks={[0, 1, 2, 3, 4]}
                        tickFormatter={(v: number) => ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"][v] || ""}
                        tick={{ fontSize: 10, fill: "#6b7280" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#fff", borderColor: "#e5e7eb", borderRadius: "12px", color: "#111827" }}
                        formatter={(value: number) => ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"][value] || value}
                      />
                      <Line type="stepAfter" dataKey="stateId" name="FSM State" stroke="#8b5cf6" strokeWidth={2.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              PANEL C — FSM State Distribution (bar chart)
              ═══════════════════════════════════════════════════════════════ */}
          {research?.fsm_session?.pct_states && Object.keys(research.fsm_session.pct_states).length > 0 && (
            <Card className="shadow-soft border-border bg-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-xl font-bold text-foreground">FSM State Distribution</CardTitle>
                <CardDescription className="text-sm text-muted-foreground">Percentage of session spent in each FSM state</CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="h-56 w-full">
                  {(() => {
                    const fsmBarData = ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"]
                      .filter((s) => research.fsm_session.pct_states[s] !== undefined)
                      .map((s) => ({ state: s, pct: research.fsm_session.pct_states[s] }));
                    const fsmColors: Record<string, string> = {
                      Baseline: "#3b82f6",
                      Monitor: "#8b5cf6",
                      Adapt: "#ef4444",
                      Recover: "#22c55e",
                      Escalate: "#f97316",
                    };
                    return (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={fsmBarData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                          <XAxis dataKey="state" tick={{ fontSize: 11, fill: "#6b7280" }} axisLine={false} tickLine={false} />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#6b7280" }} axisLine={false} tickLine={false} label={{ value: "% of Session", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 11, fill: "#6b7280" } }} />
                          <Tooltip contentStyle={{ backgroundColor: "#fff", borderColor: "#e5e7eb", borderRadius: "12px", color: "#111827" }} formatter={(value: number) => [`${value.toFixed(1)}%`, "Duration"]} />
                          <Bar dataKey="pct" radius={[6, 6, 0, 0]} maxBarSize={60}>
                            {fsmBarData.map((entry, idx) => (
                              <Cell key={idx} fill={fsmColors[entry.state] || "#94a3b8"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    );
                  })()}
                </div>
              </CardContent>
            </Card>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              PANEL D — PHQ-8 Score Card + PANEL E — Session Summary Stats
              ═══════════════════════════════════════════════════════════════ */}
          {research && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Panel D: PHQ-8 */}
              {research.phq8 && (
                <Card className="shadow-soft border-border bg-card gradient-card">
                  <CardHeader className="text-center pb-2">
                    <CardTitle className="text-2xl font-bold text-foreground">PHQ-8 Depression Screening</CardTitle>
                    <CardDescription className="text-sm text-muted-foreground">Patient Health Questionnaire-8 — Eqs. 5.1-5.3</CardDescription>
                  </CardHeader>
                  <CardContent className="text-center space-y-3 pt-4">
                    <div className={`text-6xl font-extrabold ${research.phq8.depressed_flag ? "text-destructive" : "text-green-500"}`}>
                      {research.phq8.phq8_score?.toFixed(1)}
                    </div>
                    <div className={`text-lg font-bold px-4 py-1.5 rounded-full inline-block ${
                      research.phq8.severity === "None/Minimal" ? "bg-green-500/10 text-green-600" :
                      research.phq8.severity === "Mild" ? "bg-yellow-500/10 text-yellow-600" :
                      "bg-destructive/10 text-destructive"
                    }`}>
                      {research.phq8.severity}
                    </div>
                    <div className="text-sm font-semibold px-3 py-1 rounded-full inline-block bg-muted text-muted-foreground">
                      {research.phq8.depressed_flag ? "Elevated Risk" : "Non-Depressed"}
                    </div>
                    {research.phq8.mean_stress !== undefined && (
                      <div className="flex justify-center gap-6 text-sm text-muted-foreground mt-2">
                        <span>Mean Stress: <strong>{(research.phq8.mean_stress * 100)?.toFixed(1)}%</strong></span>
                        {research.phq8.mean_confidence !== undefined && (
                          <span>Mean Confidence: <strong>{(research.phq8.mean_confidence * 100)?.toFixed(1)}%</strong></span>
                        )}
                      </div>
                    )}
                    <p className="text-xs text-muted-foreground italic mt-3 border-t pt-3">
                      Research screening estimate only — not a clinical diagnosis.
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Panel E: Session Summary Stats */}
              <Card className="shadow-soft border-border bg-card gradient-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-xl font-bold text-foreground">Session Summary</CardTitle>
                  <CardDescription className="text-sm text-muted-foreground">Multimodal behavioural metrics — Eqs. 5.4-5.15</CardDescription>
                </CardHeader>
                <CardContent className="pt-4">
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: "Duration", value: research.duration_seconds != null ? `${research.duration_seconds.toFixed(1)}s` : "—" },
                      { label: "Frames Analysed", value: research.n_frames ?? "—" },
                      { label: "Stress Mean", value: research.stress_mean != null ? `${(research.stress_mean * 100).toFixed(1)}%` : "—" },
                      { label: "Stress Max", value: research.stress_max != null ? `${(research.stress_max * 100).toFixed(1)}%` : "—" },
                      { label: "Confidence Mean", value: research.confidence_mean != null ? `${(research.confidence_mean * 100).toFixed(1)}%` : "—" },
                      { label: "Confidence Min", value: research.confidence_min != null ? `${(research.confidence_min * 100).toFixed(1)}%` : "—" },
                      { label: "Stress Spikes", value: research.bilstm_spikes?.length ?? 0 },
                      { label: "Changepoints", value: research.changepoints?.length ?? 0 },
                      { label: "FSM Transitions", value: research.fsm_session?.n_transitions ?? 0 },
                      { label: "FSM Final State", value: research.fsm_session?.final_state ?? "—" },
                    ].map(({ label, value }) => (
                      <div key={label} className="bg-secondary/40 rounded-xl px-3 py-2.5 border border-border">
                        <div className="text-xs text-muted-foreground font-medium">{label}</div>
                        <div className="text-lg font-bold text-foreground">{value}</div>
                      </div>
                    ))}
                  </div>
                  {research.pearson_r_au_cnn != null && (
                    <div className="mt-3 text-center text-sm text-muted-foreground">
                      Pearson r (AU-CNN): <strong className="text-foreground">{research.pearson_r_au_cnn.toFixed(3)}</strong>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Research loading/error indicator */}
          {researchLoading && (
            <Card className="shadow-soft border-border bg-card">
              <CardContent className="py-6 text-center text-muted-foreground animate-pulse">
                Loading research analysis from server…
              </CardContent>
            </Card>
          )}
          {researchError && !research && (
            <Card className="shadow-soft border-border bg-card">
              <CardContent className="py-6 text-center text-destructive text-sm">
                {researchError}
              </CardContent>
            </Card>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              PARTICIPANT DATASET PLOTS
              ═══════════════════════════════════════════════════════════════ */}
          {participants.length > 0 && (
            <Card className="shadow-soft border-border bg-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-xl font-bold text-foreground flex items-center gap-2">
                  <BookOpen className="h-5 w-5 text-primary" />
                  Participant Dataset Analysis
                </CardTitle>
                <CardDescription className="text-sm text-muted-foreground">
                  Stress, confidence, FSM &amp; PHQ-8 plots generated from the participant CSV datasets
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
                {/* Participant selector */}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm text-muted-foreground">Participant:</span>
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() => {
                        setSelectedParticipant("_all");
                        loadAllParticipantPlots();
                      }}
                      className={`px-3 py-1.5 rounded-full text-sm font-medium border transition-colors ${
                        selectedParticipant === "_all"
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-muted/50 text-muted-foreground border-border hover:border-primary/50"
                      }`}
                    >
                      All Participants
                    </button>
                    {participants.map((p) => (
                      <button
                        key={p.id}
                        onClick={() => {
                          setSelectedParticipant(p.id);
                          loadParticipantPlots(p.id);
                        }}
                        className={`px-3 py-1.5 rounded-full text-sm font-medium border transition-colors ${
                          selectedParticipant === p.id
                            ? "bg-primary text-primary-foreground border-primary"
                            : "bg-muted/50 text-muted-foreground border-border hover:border-primary/50"
                        }`}
                      >
                        {p.id}
                      </button>
                    ))}
                  </div>
                </div>

                {plotsLoading && (
                  <div className="py-6 text-center text-muted-foreground animate-pulse text-sm">
                    Generating plots for {selectedParticipant}…
                  </div>
                )}

                {plotsError && (
                  <div className="py-4 text-center text-destructive text-sm">{plotsError}</div>
                )}

                {!plotsLoading && plots.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {plots.map((plot) => (
                      <div key={plot.name} className="rounded-xl border border-border overflow-hidden bg-secondary/20">
                        <div className="px-3 py-2 text-xs font-semibold text-muted-foreground bg-muted/40 border-b border-border capitalize">
                          {plot.name.replace(".png", "").replace(/_/g, " ")}
                        </div>
                        <img
                          src={plot.url}
                          alt={plot.name}
                          className="w-full h-auto"
                          onError={(e) => {
                            (e.currentTarget.parentElement as HTMLElement).style.display = "none";
                          }}
                        />
                      </div>
                    ))}
                  </div>
                )}

                {!plotsLoading && !plotsError && plots.length === 0 && (
                  <div className="py-4 text-center text-xs text-muted-foreground">
                    No plots available for {selectedParticipant}.
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* INTERVIEW SESSION ANALYSIS (REST API: plots + recommendation) */}
          {(interviewSessionId || interviewResults) && (
            <Card className="shadow-soft border-border bg-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-xl font-bold text-foreground flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  Interview Session Analysis
                </CardTitle>
                <CardDescription className="text-sm text-muted-foreground">
                  4 research plots + an AI recommendation generated from the live session
                  {interviewSessionId ? ` (session ${interviewSessionId})` : ""}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-6">
                {interviewLoading && (
                  <div className="py-6 text-center text-muted-foreground animate-pulse text-sm">
                    Loading interview session results…
                  </div>
                )}
                {interviewError && (
                  <div className="py-4 text-center text-destructive text-sm">
                    {interviewError}
                  </div>
                )}
                {interviewResults && (
                  <>
                    {/* 4 session plots */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(interviewResults.plots || {}).map(([key, url]) => (
                        <div
                          key={key}
                          className="rounded-xl border border-border overflow-hidden bg-secondary/20"
                        >
                          <div className="px-3 py-2 text-xs font-semibold text-muted-foreground bg-muted/40 border-b border-border capitalize">
                            {key.replace(/_/g, " ")}
                          </div>
                          <img
                            src={url as string}
                            alt={key}
                            className="w-full h-auto"
                            onError={(e) => {
                              (e.currentTarget.parentElement as HTMLElement).style.display = "none";
                            }}
                          />
                        </div>
                      ))}
                    </div>

                    {/* Recommendation report */}
                    {interviewResults.recommendation && (
                      <div className="rounded-xl border border-border bg-secondary/10 p-5 space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <h3 className="text-lg font-bold text-foreground flex items-center gap-2">
                            <Lightbulb className="h-5 w-5 text-primary" />
                            AI Recommendation Report
                          </h3>
                          <span className="px-3 py-1 rounded-full text-xs font-semibold border border-primary/40 text-primary capitalize">
                            {interviewResults.recommendation.overall_performance?.verdict}
                          </span>
                        </div>

                        {interviewResults.recommendation.overall_performance?.summary && (
                          <p className="text-sm text-muted-foreground leading-relaxed">
                            {interviewResults.recommendation.overall_performance.summary}
                          </p>
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {interviewResults.recommendation.stress_profile && (
                            <div className="space-y-2">
                              <p className="text-sm font-semibold text-foreground">Stress Profile</p>
                              <p className="text-xs text-muted-foreground leading-relaxed">
                                {interviewResults.recommendation.stress_profile.summary}
                              </p>
                              <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                                {(interviewResults.recommendation.stress_profile.recommendations || []).map(
                                  (r: string, i: number) => (
                                    <li key={i}>{r}</li>
                                  )
                                )}
                              </ul>
                            </div>
                          )}
                          {interviewResults.recommendation.confidence_profile && (
                            <div className="space-y-2">
                              <p className="text-sm font-semibold text-foreground">Confidence Profile</p>
                              <p className="text-xs text-muted-foreground leading-relaxed">
                                {interviewResults.recommendation.confidence_profile.summary}
                              </p>
                              <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                                {(interviewResults.recommendation.confidence_profile.recommendations || []).map(
                                  (r: string, i: number) => (
                                    <li key={i}>{r}</li>
                                  )
                                )}
                              </ul>
                            </div>
                          )}
                        </div>

                        {(interviewResults.recommendation.adapting_recommendations?.length ||
                          interviewResults.recommendation.next_steps?.length) && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <p className="text-sm font-semibold text-foreground">
                                Adaptive Questioning
                              </p>
                              <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                                {(interviewResults.recommendation.adapting_recommendations || []).map(
                                  (r: string, i: number) => (
                                    <li key={i}>{r}</li>
                                  )
                                )}
                              </ul>
                            </div>
                            <div className="space-y-2">
                              <p className="text-sm font-semibold text-foreground">Next Steps</p>
                              <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                                {(interviewResults.recommendation.next_steps || []).map(
                                  (r: string, i: number) => (
                                    <li key={i}>{r}</li>
                                  )
                                )}
                              </ul>
                            </div>
                          </div>
                        )}

                        {interviewResults.recommendation.disclaimer && (
                          <p className="text-[11px] text-muted-foreground/70 border-t border-border pt-3 italic">
                            {interviewResults.recommendation.disclaimer}
                          </p>
                        )}
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {/* QUESTION BREAKDOWN */}
          <Card className="shadow-soft border-border bg-card">
            <CardHeader>
              <div className="flex items-center gap-2">
                <ListChecks className="h-6 w-6 text-primary" />
                <CardTitle className="text-xl font-bold text-foreground">Question-wise Response Breakdown</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {data.detail_per_question.map((q, i) => (
                <div key={i} className="border-b last:border-0 pb-6 last:pb-0 space-y-3">
                  <p className="font-semibold text-lg text-primary">
                    Q{i + 1}. {q.question}
                  </p>
                  <div className="bg-secondary/40 p-4 rounded-xl border border-border">
                    <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider block mb-1">Your Answer:</span>
                    <p className="text-foreground italic leading-relaxed">
                      {q.transcript?.trim()
                        ? `"${q.transcript}"`
                        : "Skipped / No response"}
                    </p>
                  </div>
                  {q.mistakes && q.mistakes !== "None" && q.mistakes.trim().length > 0 && (
                    <div className="bg-destructive/10 text-destructive p-4 rounded-xl text-sm border border-destructive/20 font-medium">
                      <span className="font-bold uppercase tracking-wider text-xs block mb-1">Feedback / Mistakes:</span> {q.mistakes}
                    </div>
                  )}
                  {q.mistakes === "None" && (
                    <div className="bg-green-500/10 text-green-600 p-4 rounded-xl text-sm border border-green-500/20 font-medium">
                      <span className="font-bold uppercase tracking-wider text-xs block mb-1">Feedback:</span> Answer looks accurate and well-structured!
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>

          {/* ACTIONS */}
          <div className="flex flex-col sm:flex-row justify-center gap-4 pt-4">


            <Button 
              size="lg" 
              variant="outline" 
              className="border-border hover:bg-secondary text-foreground font-semibold px-8 py-6 rounded-xl shadow-sm transition-all hover:scale-[1.02]" 
              onClick={() => window.print()}
            >
              Download PDF Report
            </Button>

            {(() => {
              const figSessionId = interviewSessionId || localStorage.getItem("interview_session_id") || "";
              return figSessionId ? (
                <Button
                  size="lg"
                  variant="outline"
                  className="border-border hover:bg-secondary text-foreground font-semibold px-8 py-6 rounded-xl shadow-sm transition-all hover:scale-[1.02]"
                  onClick={() => window.open(`http://localhost:8000/results/research-figure/${encodeURIComponent(figSessionId)}/download`, "_blank")}
                >
                  Download Research Figure (PNG)
                </Button>
              ) : null;
            })()}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Results;