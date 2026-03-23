import type { SpeechReport } from "./types";

// Generate timeline points for a 300-second speech
function generateTimeline(n: number) {
  const timeline = [];
  const regimes = ["Introduction", "Main Argument", "Data Presentation"];
  for (let i = 0; i < n; i++) {
    const regime = i < 83 ? regimes[0] : i < 150 ? regimes[1] : regimes[2];
    timeline.push({
      time_sec: i,
      voice: {
        syllable_rate: 3.5 + Math.sin(i / 20) * 1.5 + Math.random() * 0.5,
        pitch_cv: 0.18 + Math.sin(i / 15) * 0.08 + Math.random() * 0.05,
        emotion_label: ["Neutral", "Enthusiastic", "Nervous", "Angry", "Sad"][Math.floor(Math.random() * 5)],
        filler_count: Math.random() < 0.05 ? 1 : 0,
        disfluency_count: Math.random() < 0.08 ? 1 : 0,
      },
      body: {
        posture_score: 50 + Math.sin(i / 30) * 20 + Math.random() * 15,
        gesture_dominant: ["Active Gesture", "Rest", "Open Palm", "Rest"][Math.floor(Math.random() * 4)],
        gaze_engagement: Math.random() * 0.3,
        facial_emotion: ["Angry", "Surprise", "Neutral", "Neutral", "Happy"][Math.floor(Math.random() * 5)],
      },
      content: {
        regime_type: regime,
        grammar_error_count: Math.random() < 0.02 ? 1 : 0,
      },
    });
  }
  return timeline;
}

function generateHeatmapChannels(n: number) {
  const channels = [
    { label: "SP.RATE", key: "speech_rate" },
    { label: "PITCH", key: "pitch" },
    { label: "FILLRS", key: "fillers" },
    { label: "V.EMO", key: "voice_emotion" },
    { label: "POSTUR", key: "posture" },
    { label: "EYE CT", key: "eye_contact" },
    { label: "GESTUR", key: "gesture" },
    { label: "F.EMO", key: "face_emotion" },
    { label: "COHERE", key: "coherence" },
  ];
  return channels.map((ch) => ({
    ...ch,
    values: Array.from({ length: n }, (_, i) => {
      const base = 0.5 + Math.sin(i / 40 + channels.indexOf(ch) * 0.7) * 0.3;
      return Math.max(0.05, Math.min(1, base + (Math.random() - 0.5) * 0.3));
    }),
  }));
}

function generateEmotionArc(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    time_sec: i,
    voice_valence: Math.sin(i / 25) * 0.6 + (Math.random() - 0.5) * 0.3,
    face_valence: Math.sin(i / 30 + 1) * 0.5 + (Math.random() - 0.5) * 0.25,
    text_valence: Math.sin(i / 50) * 0.3 + 0.1,
  }));
}

export const SAMPLE_REPORT: SpeechReport = {
  id: "test_nervous",
  title: "Nervous Speaker (Student Presentation)",
  duration_sec: 300,
  processing_time_sec: 438,
  scores: {
    vocal_clarity: 82,
    body_language: 55,
    content_structure: 30,
    audience_engagement: 22,
    emotional_expressiveness: 62,
    regime_adaptability: 50,
    overall: 50,
  },
  content_details: {
    grammar_score: 0,
    fkgl: 23.5,
    ttr: 0.344,
    dominant_tone: "Informative",
    dominant_sentiment: "Neutral",
    argument_effectiveness: 25,
  },
  fusion_stats: {
    timeline_duration: 300,
    regime_boundaries: 2,
    total_disruptions: 19,
    mean_composure: 58,
    emotion_coherence: 70,
  },
  executive_summary:
    "What a pleasure to work with you today! Right off the bat, I want to celebrate your vocal clarity — you have a really strong, well-articulated delivery that comes through beautifully, and your voice carries genuine authority when you speak. There's also a lovely emotional range in your delivery that shows you're connecting with your material on a personal level.\n\nNow, as we dive into our work together, I'm seeing some wonderful opportunities to elevate your impact even further. The content you're sharing is clearly important and well-researched, but I noticed your ideas are taking some unexpected detours — think of your speech as a guided tour where each stop naturally leads to the next. Your audience engagement score tells me we can work on creating more of those magnetic moments where your listeners feel personally invested in what you're saying. The great news is that your strong vocal foundation gives us an excellent platform to build on!",
  regime_segments: [
    { regime_type: "Introduction", start_sec: 0, end_sec: 83, duration_sec: 83 },
    { regime_type: "Main Argument", start_sec: 83, end_sec: 150, duration_sec: 67 },
    { regime_type: "Data Presentation", start_sec: 150, end_sec: 300, duration_sec: 150 },
  ],
  regime_transitions: [
    { time_sec: 83, from_regime: "Introduction", to_regime: "Main Argument", score: 67 },
    { time_sec: 150, from_regime: "Main Argument", to_regime: "Data Presentation", score: 50 },
  ],
  disruption_events: [
    { time_sec: 0, timestamp: "0:00", type: "disfluency", label: "Disfluency", composure: 41, recovery_sec: 15.0, status: "SLOW" },
    { time_sec: 27, timestamp: "0:27", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 3.0, status: "GOOD" },
    { time_sec: 37, timestamp: "0:37", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 3.0, status: "GOOD" },
    { time_sec: 46, timestamp: "0:46", type: "disfluency", label: "Disfluency", composure: 41, recovery_sec: 15.0, status: "SLOW" },
    { time_sec: 46, timestamp: "0:46", type: "emotion_mismatch", label: "Emotion mismatch (voice/face)", composure: null, recovery_sec: null, status: "MISMATCH" },
    { time_sec: 52, timestamp: "0:52", type: "disfluency", label: "Disfluency", composure: 41, recovery_sec: 15.0, status: "SLOW" },
    { time_sec: 69, timestamp: "1:09", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 1.0, status: "GOOD" },
    { time_sec: 77, timestamp: "1:17", type: "emotion_mismatch", label: "Emotion mismatch (voice/face)", composure: null, recovery_sec: null, status: "MISMATCH" },
    { time_sec: 83, timestamp: "1:23", type: "transition", label: "Introduction → Main Argument", composure: null, recovery_sec: null, status: "T=67%" },
    { time_sec: 88, timestamp: "1:28", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 1.0, status: "GOOD" },
    { time_sec: 99, timestamp: "1:39", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 1.0, status: "GOOD" },
    { time_sec: 108, timestamp: "1:48", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 1.0, status: "GOOD" },
    { time_sec: 130, timestamp: "2:10", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 2.0, status: "GOOD" },
    { time_sec: 150, timestamp: "2:30", type: "transition", label: "Main Argument → Data Presentation", composure: null, recovery_sec: null, status: "T=50%" },
    { time_sec: 165, timestamp: "2:45", type: "disfluency", label: "Disfluency", composure: 80, recovery_sec: 5.0, status: "GOOD" },
    { time_sec: 195, timestamp: "3:15", type: "emotion_mismatch", label: "Emotion mismatch (voice/face)", composure: null, recovery_sec: null, status: "MISMATCH" },
    { time_sec: 220, timestamp: "3:40", type: "disfluency", label: "Disfluency", composure: 60, recovery_sec: 8.0, status: "GOOD" },
    { time_sec: 255, timestamp: "4:15", type: "disfluency", label: "Disfluency", composure: 100, recovery_sec: 1.0, status: "GOOD" },
    { time_sec: 280, timestamp: "4:40", type: "disfluency", label: "Disfluency", composure: 90, recovery_sec: 3.0, status: "GOOD" },
  ],
  coaching_feedback: [
    {
      time_sec: 0, timestamp: "0:00", type: "disruption", label: "Disfluency — rough launch",
      feedback: "Right at the very start of your speech, at 0:00, you experienced what we call a \"rough launch\" — that moment when your opening didn't quite go as planned and you found yourself momentarily thrown off course. Your composure dipped significantly in those first few seconds, and it took you about fifteen seconds to find your footing again, which felt much longer to you than it did to your audience, I'm sure.\n\nPractice your opening sentence at least twenty times out loud, but not just the words — practice the physical gesture that goes with it. Whether it's a simple hand movement, taking a step forward, or even just making direct eye contact with someone specific, pairing that opening line with a deliberate physical action creates what I call an \"anchor.\"",
    },
    {
      time_sec: 27, timestamp: "0:27", type: "disruption", label: "Disfluency — perfect recovery",
      feedback: "At the 27-second mark, you encountered a small speech stumble — maybe a word got tangled or you lost your train of thought for a moment — but here's what's remarkable: you kept your cool completely. Your composure stayed at 100%, which means your face, posture, and energy remained steady and confident.\n\nTry this: the next time you feel a disfluency coming on, lean into the moment with your audience. Take that pause, make eye contact, give a small smile.",
    },
    {
      time_sec: 37, timestamp: "0:37", type: "disruption", label: "Disfluency — smooth handling",
      feedback: "Another small stumble at 0:37, but again, your composure held at 100%. You're showing a pattern here — when disruptions happen, you handle them with grace. This is a skill that many experienced speakers struggle with, so take pride in this natural ability.",
    },
    {
      time_sec: 46, timestamp: "0:46", type: "disruption", label: "Disfluency — composure dip",
      feedback: "At 0:46, things got a bit rockier. This disfluency hit harder — your composure dropped to 41% and it took about 15 seconds to recover. What's interesting is that there was also an emotion mismatch detected at this same moment, meaning your voice and facial expressions were telling different stories.\n\nWhen you feel a bigger stumble coming, try the \"power pause\" technique: stop completely, take one deliberate breath, plant your feet, and then restart the sentence from the beginning. Your audience will read this as confidence, not confusion.",
    },
    {
      time_sec: 52, timestamp: "0:52", type: "disruption", label: "Disfluency — cascading effect",
      feedback: "Just six seconds after the previous disruption, another disfluency hit at 0:52. This suggests a cascading effect — the first stumble at 0:46 shook your confidence enough that the next one came quickly. Your composure was still at 41%, and recovery took another 15 seconds.\n\nBreak the cascade: after any noticeable stumble, physically move. Take a step to the left or right. This resets your brain's pattern and prevents the domino effect.",
    },
    {
      time_sec: 69, timestamp: "1:09", type: "disruption", label: "Disfluency — strong rebound",
      feedback: "By 1:09, you'd fully recovered from the rough patch. This disfluency was handled beautifully — 100% composure, just 1 second to recover. You bounced back stronger, which shows real resilience.",
    },
    {
      time_sec: 77, timestamp: "1:17", type: "emotion_mismatch", label: "Emotion mismatch",
      feedback: "At 1:17, your vocal emotion and facial expression weren't quite in sync. Your voice was conveying one emotion while your face told a different story. This can confuse your audience on a subconscious level.\n\nTry recording yourself and watching with the sound off, then listening without watching. Do both versions tell the same emotional story?",
    },
    {
      time_sec: 83, timestamp: "1:23", type: "transition", label: "Introduction → Main Argument",
      feedback: "At 1:23, you transitioned from your Introduction into your Main Argument. The transition scored 67% — decent but with room for improvement. Your audience needs a clear signal that you're shifting gears.\n\nUse a verbal bridge: 'Now that I've set the stage, let me take you into the heart of what I want to share.' Pair this with a physical movement — step forward or shift your position.",
    },
    {
      time_sec: 88, timestamp: "1:28", type: "disruption", label: "Disfluency — settled in",
      feedback: "A small stumble at 1:28, right after the transition. But you handled it perfectly — 100% composure, 1-second recovery. You're settling into your rhythm now.",
    },
    {
      time_sec: 99, timestamp: "1:39", type: "disruption", label: "Disfluency — consistent strength",
      feedback: "Another clean recovery at 1:39. Your composure stayed rock-solid at 100%. You're in your groove here — this is what your best speaking looks like. Remember this feeling.",
    },
    {
      time_sec: 108, timestamp: "1:48", type: "disruption", label: "Disfluency — pattern of excellence",
      feedback: "At 1:48, yet another disfluency handled with perfect composure. You're proving that the rough patch at 0:46-0:52 was the exception, not the rule. Your natural recovery ability is genuinely impressive.",
    },
    {
      time_sec: 150, timestamp: "2:30", type: "transition", label: "Main Argument → Data Presentation",
      feedback: "The transition to Data Presentation at 2:30 scored 50% — this is your weaker transition. When moving into data-heavy content, signal the shift clearly: 'Let me show you the numbers behind what I've been saying.'",
    },
  ],
  practice_plan: [
    {
      title: "THE INVISIBLE COFFEE CHAT CHALLENGE",
      description: "Transform your next practice session into a series of intimate conversations with imaginary friends scattered around the room. Pick three different spots and speak directly to each 'friend' for 30-45 seconds at a time, using their names (Sarah, Mike, Alex) and asking them direct questions like 'Sarah, have you ever felt this way?' This breaks the habit of speaking AT an audience and helps you speak WITH them.",
      frequency: "Daily, 15 minutes",
      target_dimension: "Audience Engagement",
      target_improvement: 18,
    },
    {
      title: "THE BREADCRUMB TRAIL BLUEPRINT",
      description: "Before each practice session, write your main points on sticky notes and physically arrange them on a table, then practice walking from note to note while creating verbal 'bridges' between ideas. Use transition phrases like 'Now that we've explored X, let's discover how it connects to Y.' Think of yourself as a tour guide.",
      frequency: "3x per week, 20 minutes",
      target_dimension: "Content Structure",
      target_improvement: 20,
    },
    {
      title: "THE CHAMELEON SPEAKER WORKOUT",
      description: "Take the same 3-minute story and practice delivering it to four different 'audiences' in one session: explain it to a curious 8-year-old, then to your boss in a board meeting, then to your skeptical best friend, then to your grandmother. This builds adaptability muscles.",
      frequency: "2x per week, 25 minutes",
      target_dimension: "Regime Adaptability",
      target_improvement: 10,
    },
  ],
  filler_words: [
    { time_sec: 32, timestamp: "0:32", word: "um", context: "...the reason is um because we..." },
    { time_sec: 45, timestamp: "0:45", word: "like", context: "...it was like totally..." },
    { time_sec: 138, timestamp: "2:18", word: "um", context: "...um the data shows..." },
    { time_sec: 252, timestamp: "4:12", word: "so", context: "...so basically what..." },
  ],
  transcript:
    "[0:00] Good morning everyone I am uh delighted to be here today to talk about something that matters deeply to all of us [0:05] When I first started researching this topic three years ago I had no idea how much it would change my perspective [0:12] The data we collected from over two thousand participants tells a compelling story about how we communicate and connect [0:20] But before we dive into the numbers let me share a personal experience that brought this research to life [0:27] Last summer I was presenting at a conference much like this one and I realized that the way we structure our arguments fundamentally shapes how our audience receives them [0:37] So today I want to walk you through three key findings that I believe will transform how you think about effective communication [0:46] The first finding is perhaps the most surprising and it relates to how our body language and vocal patterns interact [0:52] Research shows that speakers who maintain consistent emotional alignment between their voice and facial expressions are perceived as significantly more credible [1:00] Now this might seem obvious but the nuances are fascinating [1:09] For example when a speaker shows enthusiasm in their voice but maintains a neutral facial expression the audience subconsciously registers a disconnect [1:17] This is what we call an emotion mismatch and it can undermine even the most well-crafted argument [1:23] Moving into our main findings the data reveals three distinct patterns in effective public speaking [1:28] Pattern one successful speakers use what we call regime transitions these are deliberate shifts in their speaking style that match the content [1:39] For instance when transitioning from an anecdote to data presentation the best speakers signal this shift with both verbal and physical cues [1:48] Pattern two relates to recovery from disruptions and this is where it gets really interesting",
  timeline: generateTimeline(300),
  emotion_arc: generateEmotionArc(300),
  heatmap_channels: generateHeatmapChannels(300),
};
