import React from "react";
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import { motion } from "framer-motion";
import "./styles.css";

const DEFAULT_BULLETS = [
  "Pipeline StandardScaler ➜ LogisticRegression",
  "Validation croisée 5-fold (accuracy & F1)",
  "App Streamlit prête pour la démo",
];

const DEFAULT_METRICS = [
  { label: "Accuracy CV", value: "0.97 ± 0.02" },
  { label: "F1-macro CV", value: "0.97 ± 0.02" },
  { label: "Temps d'inférence", value: "< 10 ms" },
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
      subtitle = "Expérimente une pipeline IA complète : exploration, entraînement, évaluation et interface de prédiction.",
      highlight = "Expérience ML interactive",
      bullets = DEFAULT_BULLETS,
      metrics = DEFAULT_METRICS,
      ctaLabel = "Accéder aux prédictions",
    } = args;

    return (
      <div className="lp-root">
        <div className="lp-gradient lp-gradient-one" />
        <div className="lp-gradient lp-gradient-two" />

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
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.99 }}
                className="lp-cta"
                onClick={this.handleClick}
              >
                {ctaLabel}
              </motion.button>
              <div className="lp-cta-hint">
                <span>✨</span> Démo instantanée, aucune config requise
              </div>
            </div>

            {this.state.clicked && (
              <motion.p
                className="lp-confirmation"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                Ouverture de l’atelier de prédiction juste en dessous ↓
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
            <div className="lp-orbit">
              <div className="lp-orbit-dot" />
              <div className="lp-orbit-dot lp-orbit-dot--delay" />
            </div>
            <div className="lp-glass">
              <p className="lp-glass-eyebrow">Pipeline Iris</p>
              <h3>Logistic Regression</h3>
              <p className="lp-glass-text">
                StandardScaler → LogisticRegression(max_iter=1000) avec stratification 5-fold
              </p>
              <div className="lp-chip-row">
                <span className="lp-chip">Accuracy 97%</span>
                <span className="lp-chip">F1-macro 97%</span>
              </div>
            </div>
            <div className="lp-flow lp-flow-one" />
            <div className="lp-flow lp-flow-two" />
          </motion.div>
        </div>
      </div>
    );
  }
}

export default withStreamlitConnection(LandingComponent);
