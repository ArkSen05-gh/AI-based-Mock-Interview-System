-- Supabase SQL Schema for AI-Powered Virtual Interview System

-- 1. Enable UUID Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Sessions Table
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    domain VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'in_progress', -- in_progress, completed, failed
    confidence_overall NUMERIC(5, 2) DEFAULT 0.00,
    nervousness_overall NUMERIC(5, 2) DEFAULT 0.00,
    accuracy_overall NUMERIC(5, 2) DEFAULT 0.00,
    overall_score NUMERIC(5, 2) DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Questions Table
CREATE TABLE IF NOT EXISTS public.questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES public.sessions(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    domain VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Answers Table
CREATE TABLE IF NOT EXISTS public.answers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES public.sessions(id) ON DELETE CASCADE,
    question_id UUID REFERENCES public.questions(id) ON DELETE SET NULL,
    transcript TEXT DEFAULT '',
    audio_url TEXT,
    confidence_score NUMERIC(5, 2) DEFAULT 0.00,
    clarity_score NUMERIC(5, 2) DEFAULT 0.00,
    accuracy_score NUMERIC(5, 2) DEFAULT 0.00,
    nervousness_score NUMERIC(5, 2) DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Alerts Table (Real-time events)
CREATE TABLE IF NOT EXISTS public.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES public.sessions(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- GAZE_ALERT, STRESS_SPIKE, HIGH_NERVOUSNESS
    score NUMERIC(5, 2) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Evaluations Table (Summaries)
CREATE TABLE IF NOT EXISTS public.evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID UNIQUE REFERENCES public.sessions(id) ON DELETE CASCADE,
    feedback TEXT,
    answered_count INT DEFAULT 0,
    skipped_count INT DEFAULT 0,
    completion_rate NUMERIC(5, 2) DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Row Level Security (RLS) Policies (Supabase Specific)
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.answers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.evaluations ENABLE ROW LEVEL SECURITY;

-- Create Policies for Authenticated Users (Read/Write own data)
CREATE POLICY "Users can insert their own sessions" ON public.sessions 
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own sessions" ON public.sessions 
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own sessions" ON public.sessions 
    FOR UPDATE USING (auth.uid() = user_id);

-- Questions Policies (Linked to user's session)
CREATE POLICY "Users can handle questions for their sessions" ON public.questions
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.sessions 
            WHERE sessions.id = questions.session_id AND sessions.user_id = auth.uid()
        )
    );

-- Answers Policies (Linked to user's session)
CREATE POLICY "Users can handle answers for their sessions" ON public.answers
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.sessions 
            WHERE sessions.id = answers.session_id AND sessions.user_id = auth.uid()
        )
    );

-- Alerts Policies (Linked to user's session)
CREATE POLICY "Users can view/insert alerts for their sessions" ON public.alerts
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.sessions 
            WHERE sessions.id = alerts.session_id AND sessions.user_id = auth.uid()
        )
    );

-- Evaluations Policies (Linked to user's session)
CREATE POLICY "Users can handle evaluations for their sessions" ON public.evaluations
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.sessions 
            WHERE sessions.id = evaluations.session_id AND sessions.user_id = auth.uid()
        )
    );

-- 8. Create Performance Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_questions_session_id ON public.questions(session_id);
CREATE INDEX IF NOT EXISTS idx_answers_session_id ON public.answers(session_id);
CREATE INDEX IF NOT EXISTS idx_alerts_session_id ON public.alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_session_id ON public.evaluations(session_id);
