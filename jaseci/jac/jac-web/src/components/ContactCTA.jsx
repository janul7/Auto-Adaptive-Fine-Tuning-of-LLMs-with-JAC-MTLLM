
const FORMSPREE = "https://formspree.io/f/mqadbwgr";
import { useState } from "react";
import "../styles/contact.css";

export default function Contact() {
  const [sent, setSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);

    

     try {
      const form = e.currentTarget;
      const r = await fetch(FORMSPREE, {
        method: "POST",
        headers: { "Accept": "application/json" },
        body: new FormData(form),        // Formspree reads names: email, message, etc.
      });
      const j = await r.json();
      if (!r.ok || j.ok === false) throw new Error(j.errors?.[0]?.message || "Failed to send");
      form.reset();
      setSent(true);
    } catch (err) {
      setError(err.message || "Failed to send");
    } finally {
      setLoading(false);
    }
  }
  if (sent) {
    return (
      <section className="contact-page">
        <div className="contact-card">
          <h1 className="contact-title">Thanks! 🎉</h1>
          <p className="contact-lead">We'll reply to you soon.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="contact-page">
      <div className="contact-card">
        <header className="contact-header">
          <h1 className="contact-title">Get in touch</h1>
          <p className="contact-lead">We’d love to hear from you. Please fill out this form.</p>
        </header>

        <form className="contact-form" onSubmit={onSubmit}>
          <div className="row two">
            <label>
              <span>First name *</span>
              <input name="firstName" required />
            </label>
            <label>
              <span>Last name *</span>
              <input name="lastName" required />
            </label>
          </div>

          <label>
            <span>Email *</span>
            <input type="email" name="email" required />
          </label>

          <label>
            <span>Phone number</span>
            <input type="tel" name="phone" placeholder="+1 (555) 000-0000" />
          </label>

          <label>
            <span>Message *</span>
            <textarea name="message" rows="5" required placeholder="Leave us a message..." />
          </label>

          <label className="agree">
            <input type="checkbox" name="agree" required />
            <span>You agree to our friendly <a href="/privacy">privacy policy</a>.</span>
          </label>

          {error && <p className="status bad">{error}</p>}
          <button className="contact-submit" type="submit" disabled={loading}>
            {loading ? "Sending…" : "Send message"}
          </button>
        </form>
      </div>
    </section>
  );
}
