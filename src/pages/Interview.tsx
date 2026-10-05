import { useEffect, useState, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { Brain, Activity, Video, VideoOff, Mic, MicOff, RefreshCw, AlertTriangle, Shield, Gauge } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";


const sampleQuestions = {
  "computer-science": [
    "What are the four pillars of Object-Oriented Programming (OOP)?",
    "Explain the difference between a stack and a queue data structure.",
    "What is the time complexity of binary search?",
    "Explain the logic to reverse a string / array",
    "Difference between compiler and interpreter",
    "Describe how garbage collection works in modern programming languages.",
    "What is the difference between a process and a thread?",
    "What is a deadlock? What are the necessary conditions for deadlock?",
    "What is virtual memory?",
    "Explain what an operating system does. Name its main functions.",
  ],
  "information-technology": [
    "Explain the OSI model and its seven layers.",
    "What is data backup and recovery?",
    "What is cybersecurity?",
    "What are the key differences between symmetric and asymmetric encryption?",
    "Describe the concept of cloud computing and its service models.",
    "What is data backup and recovery?",
    "What is troubleshooting in IT?",
    "Difference between HTTP and HTTPS.",
    "Difference between SQL and NoSQL.",
    "What is troubleshooting in IT?",
  ],
  "mechanical": [
    "Explain the first law of thermodynamics.",
    "What is the difference between stress and strain?",
    "Describe the working principle of a heat exchanger.",
    "Difference between AC and DC motors (basic).",
    "What is lubrication and why is it important?",
    "What is tolerance in manufacturing?",
    "Explain refrigeration cycle.",
    "Difference between welding and soldering.",
    "What is maintenance engineering?",
    "Explain quality control methods.",
  ],
  "electrical": [
    "Explain Kirchhoff's voltage and current laws.",
    "What is the difference between AC and DC current?",
    "Describe how a transformer works.",
    "What is a circuit breaker?",
    "Difference between alternator and generator.",
    "What is earthing?",
    "What is an inductor?",
    "What is capacitance?",
    "Explain three-phase power system.",
    "What is short circuit?",
  ],
  "business-analytics": [
    "Explain the difference between descriptive and predictive analytics.",
    "What is A/B testing and when would you use it?",
    "Describe the key performance indicators (KPIs) for an e-commerce business.",
    "What is predictive analytics?",
    "What is data visualization?",
    "Explain basic statistics concepts (mean, median, mode).",
    "What is correlation?",
    "What is a dashboard?",
    "Explain business intelligence (BI).",
    "What is ROI analysis?",
  ],
  "general-business": [
    "Explain SWOT analysis and how it's used in strategic planning.",
    "What is the difference between leadership and management?",
    "Describe how you would approach a new market entry strategy.",
    "What is finance management?",
    "What is customer relationship management (CRM)?",
    "What is profit and loss statement?",
    "What is risk management?",
    "What is entrepreneurship?",
    "Difference between leadership and management.",
    "Explain supply chain management.",
  ],
};

// Universal questions
const universalQuestions = [
  "Introduce yourself.",
  "How many projects have you worked on?",
];

// Hard cap on the number of questions the interview will ask.
// Prevents the adaptive system from generating questions indefinitely.
const MAX_QUESTIONS = 8;

const Interview = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [searchParams] = useSearchParams();
  const domain = searchParams.get("domain") || "computer-science";

  const [isRecording, setIsRecording] = useState(false);
  const [videoEnabled, setVideoEnabled] = useState(true);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [questions, setQuestions] = useState<string[]>([]);
  const [loadingQuestions, setLoadingQuestions] = useState(true);
  const [transcript, setTranscript] = useState("");
  const [answers, setAnswers] = useState<string[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<any>(null);
  const audioIntervalRef = useRef<any>(null);
  const transcriptIntervalRef = useRef<any>(null);

  // Ref-based transcript tracking (replaces DOM query)
  const transcriptRef = useRef("");

  // Ref-based answers tracking (avoids stale state race in stopRecording)
  const answersRef = useRef<string[]>([]);

  // Research session
  const sessionIdRef = useRef<string>("");
  const [liveFsmState, setLiveFsmState] = useState("Baseline");
  const [liveStressPct, setLiveStressPct] = useState(0);
  const [liveConfPct, setLiveConfPct] = useState(50);

  // Refs to prevent React state stale closures in websocket/SpeechRecognition callbacks
  const isRecordingRef = useRef(false);
  const metricsHistoryRef = useRef<any[]>([]);
  const latestMetricsRef = useRef({
    eye_contact_score: 100.0,
    blink_score: 100.0,
    head_movement_score: 100.0,
    speech_rate: 0.0,
    pitch: 120.0,
    pause_duration: 0.0,
    audio_stress_score: 0.0,
    nervousness_score: 0.0,
    level: "Calm",
    stress_pct: 0.0,
    confidence_pct: 50.0,
  });

  // Keep isRecordingRef in sync
  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // ✅ Authentication check
  useEffect(() => {
    const checkAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) navigate("/auth");
    };
    checkAuth();
  }, [navigate]);

  // ✅ Load questions
  useEffect(() => {
    const fetchQuestions = async () => {
      setLoadingQuestions(true);
      try {
        const { data, error } = await supabase
          .from("questions")
          .select("*")
          .eq("domain", domain);

        let loadedQuestions = [];
        if (error || !data || data.length === 0) {
          loadedQuestions = sampleQuestions[domain] || sampleQuestions["computer-science"];
        } else {
          loadedQuestions = (data as any[]).map((q) => q.question_text || q.question);
        }

        const finalList = [...universalQuestions, ...loadedQuestions];
        setQuestions(finalList);
        const initAnswers = Array(finalList.length).fill("");
        setAnswers(initAnswers);
        answersRef.current = initAnswers;
      } catch (err) {
        console.error("⚠️ Fetch error:", err);
        const finalList = [...universalQuestions, ...(sampleQuestions[domain] || sampleQuestions["computer-science"])];
        setQuestions(finalList);
        const initAnswers = Array(finalList.length).fill("");
        setAnswers(initAnswers);
        answersRef.current = initAnswers;
      } finally {
        setLoadingQuestions(false);
      }
    };

    fetchQuestions();
  }, [domain]);

  // ✅ WebSocket & real-time video/audio streaming logic
  useEffect(() => {
    if (isRecording) {
      metricsHistoryRef.current = [];

      // Connect WebSocket
      const ws = new WebSocket("ws://localhost:8000/ws/monitor");
      wsRef.current = ws;

      ws.onopen = () => {
        // Send init message to start research session
        const sid = crypto.randomUUID();
        sessionIdRef.current = sid;
        ws.send(JSON.stringify({
          type: "init",
          session_id: sid,
          domain: domain,
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "INIT_OK") {
            sessionIdRef.current = data.session_id;
            return;
          }

          if (data.type === "METRICS_UPDATE") {
            const currentNervousness = data.fusion.score || data.fusion.nervousness_score || 0;
            latestMetricsRef.current = {
              eye_contact_score: data.face.eye_contact_score,
              blink_score: data.face.blink_score,
              head_movement_score: data.face.head_movement_score,
              speech_rate: data.voice.speech_rate,
              pitch: data.voice.pitch,
              pause_duration: data.voice.pause_duration,
              audio_stress_score: data.voice.audio_stress_score,
              nervousness_score: currentNervousness,
              level: data.fusion.level,
              stress_pct: data.research?.stress_pct || 0,
              confidence_pct: data.research?.confidence_pct || 50,
            };

            // Update live research metrics
            if (data.research) {
              setLiveStressPct(Math.round(data.research.stress_pct || 0));
              setLiveConfPct(Math.round(data.research.confidence_pct || 50));
              setLiveFsmState(data.research.fsm_state || "Baseline");
            }

            // Capture history for plotting stress spikes on the Results page
            const newPoint = {
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              stress: Math.round(currentNervousness)
            };
            metricsHistoryRef.current = [...metricsHistoryRef.current, newPoint].slice(-200);
          } else if (data.type === "ALERT") {
            toast({
              title: `${data.alert_type} Triggered`,
              description: data.message,
              variant: "destructive"
            });
          }
        } catch (err) {
          console.error("Error parsing WS message:", err);
        }
      };

      // Canvas Grabber (2 fps)
      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext("2d");

      frameIntervalRef.current = setInterval(() => {
        if (videoRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          try {
            ctx?.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            const base64Img = canvas.toDataURL("image/jpeg", 0.6);
            wsRef.current.send(JSON.stringify({
              type: "video",
              image: base64Img
            }));
          } catch (err) {
            console.error("Frame capture error:", err);
          }
        }
      }, 500);

      // Web Audio API — extract pitch, energy, speech rate in-browser
      if (streamRef.current) {
        try {
          const audioCtx = new AudioContext();
          const source = audioCtx.createMediaStreamSource(streamRef.current);
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 2048;
          source.connect(analyser);

          // Store AudioContext for cleanup
          (window as any).__interviewAudioCtx = audioCtx;

          audioIntervalRef.current = setInterval(() => {
            if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
            if (!isRecordingRef.current) return;

            const freqData = new Float32Array(analyser.frequencyBinCount);
            analyser.getFloatFrequencyData(freqData);

            const timeData = new Float32Array(analyser.fftSize);
            analyser.getFloatTimeDomainData(timeData);

            // RMS energy
            let sumSq = 0;
            for (let i = 0; i < timeData.length; i++) sumSq += timeData[i] * timeData[i];
            const rms = Math.sqrt(sumSq / timeData.length);

            // Dominant frequency (pitch proxy)
            let maxVal = -Infinity;
            let maxIdx = 0;
            for (let i = 2; i < freqData.length; i++) {
              if (freqData[i] > maxVal) { maxVal = freqData[i]; maxIdx = i; }
            }
            const sampleRate = audioCtx.sampleRate;
            const dominantFreq = (maxIdx * sampleRate) / (analyser.fftSize);

            // Speech rate: count zero-crossings as a proxy for voiced frames
            let crossings = 0;
            for (let i = 1; i < timeData.length; i++) {
              if ((timeData[i - 1] >= 0 && timeData[i] < 0) || (timeData[i - 1] < 0 && timeData[i] >= 0)) crossings++;
            }
            const speechRate = crossings / timeData.length;

            // Pitch variance: compute variance of peak frequencies over a short buffer
            // Use a simple heuristic: variance of the freq spectrum near the peak
            let pitchVariance = 0;
            if (maxIdx > 5 && maxIdx < freqData.length - 5) {
              let localMean = 0;
              const window = 10;
              for (let i = maxIdx - window; i <= maxIdx + window; i++) localMean += freqData[i];
              localMean /= (2 * window + 1);
              for (let i = maxIdx - window; i <= maxIdx + window; i++) {
                pitchVariance += (freqData[i] - localMean) ** 2;
              }
              pitchVariance = Math.sqrt(pitchVariance / (2 * window + 1)) / 100;
            }

            const audioFeatures = {
              type: "audio_features",
              pitch: Math.round(dominantFreq * 100) / 100,
              energy_rms: Math.round(rms * 10000) / 10000,
              speech_rate: Math.round(speechRate * 100) / 100,
              pitch_variance: Math.round(pitchVariance * 100) / 100,
              voice_tremor: Math.round(pitchVariance * 0.8 * 100) / 100,
            };
            wsRef.current.send(JSON.stringify(audioFeatures));
          }, 1500);
        } catch (err) {
          console.error("Web Audio API setup failed:", err);
        }
      }

      // Throttled transcript sending via ref (every 2s)
      transcriptIntervalRef.current = setInterval(() => {
        const text = transcriptRef.current;
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && text && text !== "Listening...") {
          wsRef.current.send(JSON.stringify({ type: "text", text }));
        }
      }, 2000);
    }

    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
      if (audioIntervalRef.current) {
        clearInterval(audioIntervalRef.current);
        audioIntervalRef.current = null;
      }
      if (transcriptIntervalRef.current) {
        clearInterval(transcriptIntervalRef.current);
        transcriptIntervalRef.current = null;
      }
      if ((window as any).__interviewAudioCtx) {
        try { (window as any).__interviewAudioCtx.close(); } catch (e) {}
        (window as any).__interviewAudioCtx = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [isRecording, toast]);

  // ✅ Timer logic
  useEffect(() => {
    let interval: any;
    if (isRecording) {
      interval = setInterval(() => setTimeElapsed((prev) => prev + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  // ✅ Setup camera
  const setupCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
      streamRef.current = stream;
    } catch (error) {
      console.error("Camera error:", error);
      toast({
        title: "Camera Permission Denied",
        description: "Please allow camera and microphone access to continue.",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    setupCamera();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (recognitionRef.current) recognitionRef.current.stop();
    };
  }, []);

  const toggleVideo = () => {
    if (streamRef.current) {
      const videoTrack = streamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setVideoEnabled(videoTrack.enabled);
      }
    }
  };

  const toggleAudio = () => {
    if (streamRef.current) {
      const audioTrack = streamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setAudioEnabled(audioTrack.enabled);
      }
    }
  };

  const saveCurrentAnswer = () => {
    setAnswers((prev) => {
      const copy = [...prev];
      copy[currentQuestion] = transcript || "";
      answersRef.current = copy;
      return copy;
    });
  };

  // ✅ Speech Recognition
  const startRecording = () => {
    setIsRecording(true);
    setTranscript("");
    transcriptRef.current = "";

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.continuous = true;
      recognition.interimResults = true;

      // Track finalized text across restarts
      let finalizedTextBeforeRestart = "";

      recognition.onresult = (event: any) => {
        let finalTranscript = "";
        let interimTranscript = "";
        for (let i = 0; i < event.results.length; i++) {
          const transcriptSegment = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcriptSegment + " ";
          } else {
            interimTranscript += transcriptSegment;
          }
        }
        const combined = (finalizedTextBeforeRestart + finalTranscript + interimTranscript).trim();
        setTranscript(combined);
        transcriptRef.current = combined;
      };

      recognition.onerror = (err: any) => {
        console.error("Speech recognition error:", err);
      };

      recognition.onend = () => {
        // If we are still recording, restart the recognition to keep it alive
        if (isRecordingRef.current) {
          // Save what we have so far as finalized before the new session begins
          finalizedTextBeforeRestart = transcriptRef.current + " ";
          try {
            recognition.start();
          } catch (e) {
            console.error("Failed to restart speech recognition:", e);
          }
        }
      };

      try {
        recognition.start();
        recognitionRef.current = recognition;
      } catch (e) {
        console.error("Failed to start speech recognition:", e);
      }
    } else {
      toast({
        title: "Speech Recognition Unavailable",
        description: "Your browser does not support speech recognition. Try Chrome.",
        variant: "destructive",
      });
    }

    toast({
      title: "Recording Started",
      description: "Your interview session has begun. Good luck!",
    });
  };

  // ✅ Stop recording + send to backend
  const stopRecording = async () => {
    setIsRecording(false);
    isRecordingRef.current = false;
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      } catch (e) {}
    }
    saveCurrentAnswer();

    // Close WebSocket explicitly and wait for server to finalize the session
    const ws = wsRef.current;
    if (ws) {
      wsRef.current = null;
      // Send a finalize message before closing
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "end" }));
      }
      // Wait for close
      await new Promise<void>((resolve) => {
        ws.onclose = () => resolve();
        ws.onerror = () => resolve();
        setTimeout(() => { try { ws.close(); } catch (e) {} resolve(); }, 2000);
      });
    }
    // Give server 1 second to persist research results
    await new Promise((r) => setTimeout(r, 1000));

    const payload = {
      answers: questions.map((q, idx) => ({
        question: q,
        answer: answersRef.current[idx] || (idx === currentQuestion ? transcriptRef.current : ""),
      })),
      domain,
      session_id: sessionIdRef.current,
    };

    const formData = new FormData();
    formData.append("answers_json", JSON.stringify(payload));

    toast({
      title: "Interview Ended",
      description: "Please wait, analyzing your performance...",
    });

    try {
      const res = await fetch("http://localhost:8000/evaluate_multimodal", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        data.nervousness_index = latestMetricsRef.current.nervousness_score;
        data.live_metrics = {
          stress_pct: latestMetricsRef.current.stress_pct,
          confidence_pct: latestMetricsRef.current.confidence_pct,
          fsm_state: liveFsmState,
        };
        localStorage.setItem("last_evaluation", JSON.stringify(data));
        localStorage.setItem("realtime_stress_history", JSON.stringify(metricsHistoryRef.current));
        // Persist session_id so Results page can fetch the research summary
        if (sessionIdRef.current) {
          localStorage.setItem("interview_session_id", sessionIdRef.current);
        }

        toast({
          title: "Evaluation Complete",
          description: "Your interview has been successfully analyzed!",
        });
        // Pass the session id to the Results page via React router state
        // (no localStorage/sessionStorage). The Results page uses it to fetch
        // the full /interview/results payload (plots + metrics + report).
        const sessionId = sessionIdRef.current || "";
        setTimeout(() => navigate("/results", { state: { sessionId } }), 1200);
      } else {
        toast({
          title: "Evaluation Failed",
          description: "Server error while analyzing your interview.",
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error("Evaluation error:", err);
      toast({
        title: "Connection Error",
        description: "Could not reach backend server.",
        variant: "destructive",
      });
    }
  };

  const nextQuestion = async () => {
    if (currentQuestion >= questions.length - 1) {
      toast({
        title: "Interview Complete",
        description: "You have answered all questions. End the interview to see your results.",
      });
      return;
    }

    saveCurrentAnswer();
    const previousTranscript = transcript;

    setTranscript("");
    transcriptRef.current = ""; // Clear ref for next question
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop(); // Stops current question speech recognition session
      } catch (e) {
        console.error(e);
      }
    }

    // Only ask the adaptive engine for a new question if we haven't hit the cap.
    if (questions.length < MAX_QUESTIONS) {
      try {
        const res = await fetch("http://localhost:8000/next_question", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            domain: domain,
            previous_answer: previousTranscript,
            history: [],
            session_id: sessionIdRef.current,
            question_count: questions.length,
          })
        });

        if (res.ok) {
          const data = await res.json();
          if (data.question && questions.length < MAX_QUESTIONS) {
            setQuestions((prev) => {
              const copy = [...prev];
              copy.splice(currentQuestion + 1, 0, data.question);
              return copy;
            });
            setAnswers((prev) => {
              const copy = [...prev];
              copy.splice(currentQuestion + 1, 0, "");
              answersRef.current = copy;
              return copy;
            });
          }
        }
      } catch (e) {
        console.error("Failed to fetch dynamic question", e);
      }
    }

    setCurrentQuestion((prev) => prev + 1);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // Derived display variables
  const questionNumber = currentQuestion + 1;
  const totalQuestions = questions.length;
  const currentQuestionText = questions[currentQuestion] || "";

  // ✅ Skip question
  const skipQuestion = () => {
    if (currentQuestion >= questions.length - 1) {
      toast({
        title: "Interview Complete",
        description: "You have answered all questions. End the interview to see your results.",
      });
      return;
    }
    setTranscript("");
    transcriptRef.current = ""; // Clear ref for skipped question
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop(); // Reset buffer for the next question
      } catch (e) {
        console.error(e);
      }
    }
    setCurrentQuestion((prev) => Math.min(prev + 1, questions.length - 1));
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <header className="border-b bg-card/80 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary animate-pulse" />
            <h1 className="text-2xl font-bold text-foreground">InterviewIQ</h1>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {/* Research live metrics */}
            {isRecording && (
              <>
                <div className="flex items-center gap-1.5 text-xs font-semibold bg-secondary/80 px-2.5 py-1 rounded-full">
                  <Shield className="h-3.5 w-3.5 text-violet-500" />
                  <span className="text-muted-foreground">FSM:</span>
                  <span className={`font-bold ${liveFsmState === "Adapt" || liveFsmState === "Escalate" ? "text-amber-500" : liveFsmState === "Monitor" ? "text-blue-500" : "text-green-500"}`}>
                    {liveFsmState}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-xs font-semibold bg-secondary/80 px-2.5 py-1 rounded-full">
                  <Gauge className="h-3.5 w-3.5 text-red-500" />
                  <span className="text-muted-foreground">Stress:</span>
                  <span className={`font-bold ${liveStressPct > 65 ? "text-red-500" : liveStressPct > 40 ? "text-amber-500" : "text-green-500"}`}>
                    {liveStressPct}%
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-xs font-semibold bg-secondary/80 px-2.5 py-1 rounded-full">
                  <Activity className="h-3.5 w-3.5 text-blue-500" />
                  <span className="text-muted-foreground">Conf:</span>
                  <span className={`font-bold ${liveConfPct < 35 ? "text-red-500" : liveConfPct < 60 ? "text-amber-500" : "text-green-500"}`}>
                    {liveConfPct}%
                  </span>
                </div>
              </>
            )}
            <div className="text-sm text-muted-foreground font-semibold bg-secondary px-3 py-1.5 rounded-full">
              Elapsed: {formatTime(timeElapsed)}
            </div>
          </div>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8 flex flex-col lg:flex-row flex-1 gap-8 items-stretch">
        {/* Left column – video feed and media toggles */}
        <section className="w-full lg:w-1/2 flex flex-col gap-6 bg-card rounded-2xl p-6 border border-border shadow-soft">
          <div className="flex flex-wrap gap-4 items-center justify-between">
            <div className="flex gap-2">
              <button
                onClick={toggleVideo}
                className={`p-3 rounded-xl border transition-all shadow-sm ${videoEnabled ? "bg-background border-border text-foreground hover:bg-secondary" : "bg-destructive/10 border-destructive/20 text-destructive hover:bg-destructive/20"}`}
              >
                {videoEnabled ? <Video className="h-5 w-5" /> : <VideoOff className="h-5 w-5" />}
              </button>
              <button
                onClick={toggleAudio}
                className={`p-3 rounded-xl border transition-all shadow-sm ${audioEnabled ? "bg-background border-border text-foreground hover:bg-secondary" : "bg-destructive/10 border-destructive/20 text-destructive hover:bg-destructive/20"}`}
              >
                {audioEnabled ? <Mic className="h-5 w-5" /> : <MicOff className="h-5 w-5" />}
              </button>
            </div>
            {!isRecording ? (
              <button
                onClick={() => {
                  const hasVideo = streamRef.current && streamRef.current.getVideoTracks().some(t => t.enabled);
                  const hasAudio = streamRef.current && streamRef.current.getAudioTracks().some(t => t.enabled);
                  if (!hasVideo || !hasAudio || !videoEnabled || !audioEnabled) {
                    toast({
                      title: "Enable Media",
                      description: "Please enable both webcam and microphone access before starting the interview.",
                      variant: "destructive",
                    });
                    return;
                  }
                  startRecording();
                }}
                className="px-6 py-3 bg-primary hover:bg-primary/95 text-primary-foreground font-semibold rounded-xl shadow-md flex items-center gap-2 transition-all hover:scale-[1.02]"
              >
                <Activity className="h-5 w-5" /> Start Interview
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="px-6 py-3 bg-destructive hover:bg-destructive/90 text-destructive-foreground font-semibold rounded-xl shadow-md flex items-center gap-2 transition-all hover:scale-[1.02]"
              >
                <AlertTriangle className="h-5 w-5" /> End Interview
              </button>
            )}
          </div>
          <div className="relative aspect-video bg-muted border border-border rounded-2xl overflow-hidden shadow-inner flex-1 min-h-[300px]">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover transition-opacity duration-300 ${videoEnabled ? "opacity-100" : "opacity-0"}`}
            />
            {!videoEnabled && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground bg-muted/50">
                <VideoOff className="h-12 w-12 text-muted-foreground/60 animate-pulse" />
                <span className="mt-2 font-medium">Video Disabled</span>
              </div>
            )}
            
            {/* Live Indicator overlay if recording */}
            {isRecording && (
              <div className="absolute top-4 right-4 bg-red-600 text-white text-xs font-bold px-3 py-1 rounded-full flex items-center gap-2 animate-pulse shadow">
                <span className="h-2 w-2 rounded-full bg-white animate-ping"></span>
                LIVE ANALYSIS
              </div>
            )}
          </div>
        </section>

        {/* Right column – question, transcript, navigation */}
        <section className="w-full lg:w-1/2 flex flex-col gap-6">
          <div className="p-6 bg-card border border-border rounded-2xl shadow-soft gradient-card flex-1 flex flex-col justify-center min-h-[150px]">
            <h3 className="text-lg font-bold text-foreground mb-3 flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              Question {questionNumber} of {totalQuestions}
            </h3>
            <p className="text-foreground text-xl leading-relaxed font-semibold">{currentQuestionText || "Click 'Start Interview' to begin."}</p>
          </div>
          
          <div className="p-6 bg-card border border-border rounded-2xl h-64 overflow-y-auto shadow-inner flex flex-col justify-between">
            <div className="space-y-2">
              <span className="text-xs font-semibold text-primary uppercase tracking-wider">Live Transcript</span>
              <p className="text-base text-foreground/80 mt-1 leading-relaxed italic">
                {transcript || (isRecording ? "Listening..." : "Your speech transcript will display here after you start speaking.")}
              </p>
            </div>
          </div>
          
          <div className="flex gap-4 justify-end mt-2">
            <button
              onClick={skipQuestion}
              disabled={!isRecording}
              className="px-6 py-3.5 bg-secondary hover:bg-secondary/80 text-foreground font-semibold rounded-xl border border-border transition-all disabled:opacity-50"
            >
              Skip Question
            </button>
            <button
              onClick={nextQuestion}
              disabled={!isRecording}
              className="px-6 py-3.5 bg-primary hover:bg-primary/95 text-primary-foreground font-semibold rounded-xl shadow-md transition-all hover:scale-[1.02] disabled:opacity-50"
            >
              Next Question
            </button>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Interview;
