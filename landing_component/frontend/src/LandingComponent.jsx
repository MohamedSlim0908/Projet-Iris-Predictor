import React from "react";
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import { motion } from "framer-motion";
import "./styles.css";

const DEFAULT_BULLETS = [
  "Glance at the Iris dataset through a couple of easy-to-read charts",
  "Follow the learning process with a short, step-by-step explanation",
  "Try a prediction without knowing anything about machine learning",
];

const DEFAULT_METRICS = [
  { label: "Accuracy CV", value: "0.97 ± 0.02" },
  { label: "F1-macro CV", value: "0.97 ± 0.02" },
  { label: "Prediction time", value: "< 10 ms" },
];

const JOURNEY_STEPS = [
  {
    title: "Explore",
    desc: "Start from the Iris dataset and highlight the main trends visually.",
  },
  {
    title: "Validate",
    desc: "Train the model, check the metrics, and compare with hold-out samples.",
  },
  {
    title: "Share",
    desc: "Use the Streamlit interface to demo the solution like a real product.",
  },
];

const STUDIO_FEATURES = [
  {
    title: "Guided discovery",
    badge: "Explore",
    desc: "A handful of charts is all you need to understand what separates each species.",
  },
  {
    title: "One-click actions",
    badge: "Scripts",
    desc: "Train, evaluate, or infer from the command line without extra setup.",
  },
  {
    title: "Live interface",
    badge: "Demo",
    desc: "Type the measurements and the page returns the most likely Iris species.",
  },
];

class LandingComponent extends StreamlitComponentBase {
  constructor(props) {
    super(props);
    this.state = { clicked: false };
    this.handleClick = this.handleClick.bind(this);
  }

  componentDidMount() {
    Streamlit.setFrameHeight();
  }

  componentDidUpdate() {
    Streamlit.setFrameHeight();
  }

  handleClick() {
    this.setState({ clicked: true });
    Streamlit.setComponentValue(true);
  }

  render() {
    const args = this.props.args || {};
    const {
      title = "Iris Predictor Studio",
      subtitle =
        "Follow the Iris journey even without a data background: understand the data, watch the"
        + " training process, then try a prediction in two clicks.",
      highlight = "Guided tour",
      bullets = DEFAULT_BULLETS,
      metrics = DEFAULT_METRICS,
      ctaLabel = "Open the prediction form",
    } = args;

    return (
      <div className="lp-root">
        <div className="lp-gradient lp-gradient-one" />
        <div className="lp-gradient lp-gradient-two" />
        <motion.nav
          className="lp-nav"
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="lp-logo">Iris Predictor</div>
          <div className="lp-nav-badges">
            <span className="lp-nav-pill">LogReg Pipeline</span>
            <span className="lp-nav-pill lp-nav-pill--ghost">Streamlit + React</span>
          </div>
        </motion.nav>

        <div className="lp-shell">
          <motion.div
            className="lp-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="lp-pill">{highlight}</div>
            <motion.h1
              className="lp-title"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.6 }}
            >
              {title}
            </motion.h1>

            <motion.p
              className="lp-subtitle"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              {subtitle}
            </motion.p>

            <ul className="lp-list">
              {(bullets.length ? bullets : DEFAULT_BULLETS).map((item, index) => (
                <motion.li
                  key={item}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + index * 0.1, duration: 0.4 }}
                >
                  <span className="lp-dot" />
                  {item}
                </motion.li>
              ))}
            </ul>

            <div className="lp-cta-row">
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                className="lp-cta"
                onClick={this.handleClick}
              >
                {ctaLabel}
              </motion.button>
              <div className="lp-cta-hint">
                <span>✨</span> Instant demo — no installation needed
              </div>
            </div>

            {this.state.clicked && (
              <motion.p
                className="lp-confirmation"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                The prediction form is now available just below ↓
              </motion.p>
            )}

            <div className="lp-metrics">
              {(metrics.length ? metrics : DEFAULT_METRICS).map((metric) => (
                <motion.div
                  className="lp-metric-card"
                  key={metric.label}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                >
                  <span className="lp-metric-value">{metric.value}</span>
                  <span className="lp-metric-label">{metric.label}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>

          <motion.div
            className="lp-visual"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            <div className="lp-visual-radar">
              <div className="lp-radar-ring lp-radar-ring--one" />
              <div className="lp-radar-ring lp-radar-ring--two" />
              <div className="lp-radar-ring lp-radar-ring--three" />
            </div>

            <div className="lp-glass">
              <p className="lp-glass-eyebrow">Pipeline Iris</p>
              <h3>Logistic Regression</h3>
              <p className="lp-glass-text">
                We clean the measurements, standardize the data, and train a reliable logistic
                regression to separate the three species.
              </p>
              <div className="lp-chip-row">
                <span className="lp-chip">Accuracy 97%</span>
                <span className="lp-chip">F1-macro 97%</span>
                <span className="lp-chip">Hold-out 93%</span>
              </div>
            </div>

            <div className="lp-feature-grid">
              {STUDIO_FEATURES.map((feature, idx) => (
                <motion.div
                  key={feature.title}
                  className="lp-feature-card"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + idx * 0.1 }}
                >
                  <span className="lp-feature-badge">{feature.badge}</span>
                  <h4>{feature.title}</h4>
                  <p>{feature.desc}</p>
                </motion.div>
              ))}
            </div>

            <div className="lp-flow lp-flow-one" />
            <div className="lp-flow lp-flow-two" />
          </motion.div>
        </div>

        <div className="lp-ribbon">
          <motion.div
            className="lp-ribbon-label"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            Full workflow
          </motion.div>
          <div className="lp-ribbon-track">
            {JOURNEY_STEPS.map((step, index) => (
              <motion.div
                key={step.title}
                className="lp-timeline-step"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="lp-step-index">0{index + 1}</div>
                <div>
                  <h5>{step.title}</h5>
                  <p>{step.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    );
  }
}

export default withStreamlitConnection(LandingComponent);
