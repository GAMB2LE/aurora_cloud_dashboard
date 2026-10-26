"""Process-wide component classes shared by all dashboard browser sessions.

Panel executes app.py afresh for each session. Defining ReactiveHTML or
JSComponent subclasses there registers new classes on every request. Panel
then enumerates the accumulated classes when bundling page resources. Keep
class definitions in this normally imported module; widget instances and
callbacks remain owned by their individual sessions.
"""

import panel as pn
import param


class OverviewSnapshotHTML(pn.custom.JSComponent):
    """Age the charger locally, even if the snapshot websocket disconnects."""

    object = param.String(default="")

    _esm = """
    export function ageCharger(card, now = Date.now()) {
      const stamp = Date.parse(card.dataset.chargerSample);
      const age = (now - stamp) / 60000;
      const value = card.querySelector('.mobile-status-card__value');
      const meta = card.querySelector('.mobile-status-card__meta');
      if (!Number.isFinite(age) || age < 0 || age > 30) {
        value.textContent = 'Unknown';
        card.classList.remove('mobile-status-card--green');
        card.classList.add('mobile-status-card--gray');
        meta.textContent = age > 30 ? `Last sample ${Math.floor(age)}m ago`
          : 'Charging telemetry unavailable';
      } else if (value.textContent !== 'Unknown') {
        const power = meta.textContent.split(' · ')[0];
        meta.textContent = `${power} · Sample ${Math.floor(age)}m ago`;
      }
    }

    export function render({ model }) {
      const content = document.createElement('div');
      const age = () => content.querySelectorAll('[data-charger-sample]')
        .forEach(card => ageCharger(card));
      const update = () => {
        content.innerHTML = model.object;
        age();
      };
      update();
      model.on('change:object', update);
      const timer = setInterval(age, 60000);
      document.addEventListener('visibilitychange', age);
      window.addEventListener('pageshow', age);
      model.on('remove', () => {
        clearInterval(timer);
        document.removeEventListener('visibilitychange', age);
        window.removeEventListener('pageshow', age);
      });
      return content;
    }
    """


_BROWSER_PERF_EVENTS = {
    "browser_document_ready",
    "browser_first_power_plot",
    "browser_power_section_switch",
}


class WxcamVideoPlayer(pn.reactive.ReactiveHTML):
    src = param.String(default="")
    poster = param.String(default="")
    title = param.String(default="")
    subtitle = param.String(default="")
    mode_class = param.String(default="wxcam-player--wide")

    _template = """
    <div id="player_shell" class="wxcam-player ${mode_class}">
      <div id="meta_row" class="wxcam-player__meta">
        <div id="meta_text" class="wxcam-player__meta-text">
          <div id="title_text" class="wxcam-player__title">{{ title }}</div>
          <div id="subtitle_text" class="wxcam-player__subtitle">{{ subtitle }}</div>
        </div>
      </div>
      <div id="control_row" class="wxcam-player__controls">
        <button id="play_btn" type="button" onclick="${script('toggle_play')}">Play</button>
        <button id="back_btn" type="button" onclick="${script('jump_back')}">-0.5s</button>
        <button id="forward_btn" type="button" onclick="${script('jump_forward')}">+0.5s</button>
        <label id="speed_wrap" class="wxcam-player__inline-label">
          <span>Speed</span>
          <select id="speed_select" onchange="${script('change_speed')}">
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </label>
        <label id="loop_wrap" class="wxcam-player__inline-label wxcam-player__checkbox">
          <input id="loop_toggle" type="checkbox" onchange="${script('toggle_loop')}"></input>
          <span>Loop</span>
        </label>
      </div>
      <div id="seek_row" class="wxcam-player__seek">
        <span id="current_time" class="wxcam-player__time">00:00</span>
        <input id="seek_slider" type="range" min="0" max="1000" value="0" step="1" oninput="${script('seek')}"></input>
        <span id="duration" class="wxcam-player__time">00:00</span>
      </div>
      <div id="video_frame" class="wxcam-player__frame">
        <video
          id="video_el"
          src="${src}"
          poster="${poster}"
          controls
          preload="metadata"
          playsinline
          onloadedmetadata="${script('sync_metadata')}"
          ontimeupdate="${script('sync_time')}"
          onplay="${script('sync_play_state')}"
          onpause="${script('sync_play_state')}"
          onended="${script('sync_play_state')}"
          onratechange="${script('sync_speed_state')}"
        ></video>
      </div>
    </div>
    """

    _scripts = {
        "src": """
          video_el.pause();
          video_el.load();
          seek_slider.value = 0;
          current_time.textContent = "00:00";
          duration.textContent = "00:00";
          play_btn.textContent = "Play";
          speed_select.value = "1";
        """,
        "toggle_play": """
          if (video_el.paused) {
            video_el.play();
          } else {
            video_el.pause();
          }
        """,
        "jump_back": """
          const nextTime = Math.max(0, (video_el.currentTime || 0) - 0.5);
          video_el.currentTime = nextTime;
          view.run_script('sync_time');
        """,
        "jump_forward": """
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          const candidateTime = (video_el.currentTime || 0) + 0.5;
          const nextTime = durationSeconds > 0 ? Math.min(durationSeconds, candidateTime) : candidateTime;
          video_el.currentTime = nextTime;
          view.run_script('sync_time');
        """,
        "change_speed": """
          video_el.playbackRate = Number(speed_select.value || 1);
          view.run_script('sync_speed_state');
        """,
        "toggle_loop": """
          video_el.loop = loop_toggle.checked;
        """,
        "seek": """
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          if (!durationSeconds) {
            return;
          }
          video_el.currentTime = (Number(seek_slider.value || 0) / 1000) * durationSeconds;
          view.run_script('sync_time');
        """,
        "sync_metadata": """
          const formatTime = (seconds) => {
            if (!Number.isFinite(seconds) || seconds < 0) {
              return "00:00";
            }
            const total = Math.floor(seconds);
            const hours = Math.floor(total / 3600);
            const minutes = Math.floor((total % 3600) / 60);
            const secs = total % 60;
            if (hours > 0) {
              return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
          };
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          current_time.textContent = formatTime(video_el.currentTime || 0);
          duration.textContent = formatTime(durationSeconds);
          seek_slider.value = 0;
          video_el.loop = loop_toggle.checked;
          video_el.playbackRate = Number(speed_select.value || 1);
          view.run_script('sync_play_state');
          view.run_script('sync_speed_state');
        """,
        "sync_time": """
          const formatTime = (seconds) => {
            if (!Number.isFinite(seconds) || seconds < 0) {
              return "00:00";
            }
            const total = Math.floor(seconds);
            const hours = Math.floor(total / 3600);
            const minutes = Math.floor((total % 3600) / 60);
            const secs = total % 60;
            if (hours > 0) {
              return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
          };
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          const currentSeconds = Number(video_el.currentTime || 0);
          current_time.textContent = formatTime(currentSeconds);
          duration.textContent = formatTime(durationSeconds);
          seek_slider.value = durationSeconds > 0 ? Math.round((currentSeconds / durationSeconds) * 1000) : 0;
        """,
        "sync_play_state": """
          play_btn.textContent = video_el.paused ? "Play" : "Pause";
        """,
        "sync_speed_state": """
          speed_select.value = String(video_el.playbackRate || 1);
        """,
    }


class BrowserPerformanceProbe(pn.custom.JSComponent):
    """Report real browser milestones to the development performance log."""

    message = param.Dict(default={})

    _esm = """
    export function render({ model }) {
      const marker = document.createElement("span");
      marker.hidden = true;
      marker.setAttribute("aria-hidden", "true");

      const roots = () => {
        const found = [document];
        for (let index = 0; index < found.length; index += 1) {
          const root = found[index];
          for (const element of root.querySelectorAll("*")) {
            if (element.shadowRoot && !found.includes(element.shadowRoot)) {
              found.push(element.shadowRoot);
            }
          }
        }
        return found;
      };
      const plots = () => roots().flatMap(
        (root) => Array.from(root.querySelectorAll(".js-plotly-plot"))
      ).filter((node) => node.getClientRects().length > 0);
      const plotSignature = () => {
        const nodes = plots();
        // Plotly stores trace metadata on the SVG renderer but not on the
        // WebGL canvas renderer. Use the visible axis/title text as a stable
        // render signature so Current and Forecast remain distinguishable.
        const labels = nodes.map((node) => Array.from(node.querySelectorAll("text"))
          .map((text) => text.textContent.trim())
          .filter(Boolean)
          .join("|"))
          .join("||");
        return `${nodes.length}:${labels}`;
      };
      const navigation = performance.getEntriesByType("navigation")[0] || {};
      const emit = (event, duration, extra = {}) => {
        const payload = {
          event,
          duration_ms: Math.round(Number(duration) * 1000) / 1000,
          path: `${location.pathname}${location.search}`,
          viewport_width: window.innerWidth,
          viewport_height: window.innerHeight,
          ...extra,
        };
        window.setTimeout(() => {
          model.message = { ...payload, nonce: Date.now() };
        }, 0);
      };

      // This component itself arrives with the first Bokeh document, so a
      // paint frame is sufficient to observe the shell. The former one-second
      // timeout inflated the reported shell latency without delaying the UI.
      const reportDocumentReady = () => emit("browser_document_ready", performance.now(), {
          response_start_ms: Number(navigation.responseStart || 0),
          dom_interactive_ms: Number(navigation.domInteractive || 0),
          load_event_end_ms: Number(navigation.loadEventEnd || 0),
          navigation_type: String(navigation.type || "unknown"),
        });
      if (typeof window.requestAnimationFrame === "function") {
        window.requestAnimationFrame(reportDocumentReady);
      } else {
        window.setTimeout(reportDocumentReady, 0);
      }

      let firstPlotSent = false;
      let firstPlotTimer = null;
      const checkFirstPlot = () => {
        if (firstPlotSent) return;
        if (firstPlotTimer === null) {
          const started = performance.now();
          firstPlotTimer = window.setInterval(() => {
          const nodes = plots();
          if (!nodes.length && performance.now() - started < 20000) return;
          window.clearInterval(firstPlotTimer);
          firstPlotTimer = null;
          if (!nodes.length) return;
          firstPlotSent = true;
          emit("browser_first_power_plot", performance.now(), {
            plot_count: nodes.length,
            renderer: nodes.some((node) => node.querySelector("canvas")) ? "webgl" : "svg",
          });
          }, 100);
        }
      };

      let switchTimer = null;
      const onClick = (event) => {
        // Panel's button group is rendered in a shadow root. ``closest`` on
        // the retargeted event node cannot see that button, whereas the
        // composed path preserves the original element for this measurement.
        const button = event.composedPath().find(
          (node) => node instanceof HTMLElement && node.matches("button")
        );
        if (!button) return;
        const label = button.textContent.trim();
        if (label !== "Current Conditions" && label !== "Forecast & Planning") return;
        if (switchTimer !== null) window.clearInterval(switchTimer);
        const started = performance.now();
        const original = plotSignature();
        switchTimer = window.setInterval(() => {
          const elapsed = performance.now() - started;
          const signature = plotSignature();
          if (signature === original && elapsed < 20000) return;
          window.clearInterval(switchTimer);
          switchTimer = null;
          emit("browser_power_section_switch", elapsed, {
            section: label,
            plot_signature_before: original,
            plot_signature_after: signature,
            timed_out: elapsed >= 20000,
          });
        }, 100);
      };

      document.addEventListener("click", onClick, true);
      // Panel nests figures inside several shadow roots. Poll briefly while
      // the initial document hydrates; a document-only MutationObserver does
      // not observe mutations made inside those nested component roots.
      checkFirstPlot();

      marker.remove = (() => {
        const remove = marker.remove.bind(marker);
        return () => {
          document.removeEventListener("click", onClick, true);
          if (firstPlotTimer !== null) window.clearTimeout(firstPlotTimer);
          if (switchTimer !== null) window.clearInterval(switchTimer);
          remove();
        };
      })();
      return marker;
    }
    """

    def __init__(self, *, log_callback=None, **params) -> None:
        super().__init__(**params)
        self._log_callback = log_callback
        self.param.watch(self._handle_param_message, "message")

    def _handle_param_message(self, event) -> None:
        self._handle_msg(event.new)

    def _handle_msg(self, data) -> None:
        if not isinstance(data, dict):
            return
        event = str(data.get("event", ""))
        if event not in _BROWSER_PERF_EVENTS:
            return
        try:
            duration_ms = float(data.get("duration_ms"))
        except (TypeError, ValueError):
            return
        fields = {
            str(key): value
            for key, value in data.items()
            if key not in {"event", "duration_ms"}
        }
        if self._log_callback is not None:
            self._log_callback(event, duration_ms=duration_ms, instrument="power", **fields)


class WxcamSelectionState(param.Parameterized):
    selected_hour_path = param.String(default="")


class AuroracamCameraCard(pn.reactive.ReactiveHTML):
    camera_id = param.String(default="")
    title = param.String(default="")
    ip = param.String(default="")
    image_url = param.String(default="")
    image_class = param.String(default="auroracam-card__hidden")
    placeholder = param.String(default="No image")
    placeholder_class = param.String(default="")
    meta = param.String(default="")
    filename = param.String(default="")
    selected_class = param.String(default="")
    empty_class = param.String(default="auroracam-card--empty")
    clicked = param.Event()

    _template = """
    <button id="card" type="button" class="auroracam-card auroracam-card--button {{ selected_class }} {{ empty_class }}" onclick="${script('select')}">
      <div class="auroracam-card__head">
        <div class="auroracam-card__title">{{ title }}</div>
        <div class="auroracam-card__ip">{{ ip }}</div>
      </div>
      <img class="auroracam-card__img {{ image_class }}" src="{{ image_url }}" alt="{{ title }} selected image"></img>
      <div class="auroracam-card__placeholder {{ placeholder_class }}">{{ placeholder }}</div>
      <div class="auroracam-card__meta">{{ meta }}</div>
      <div class="auroracam-card__file">{{ filename }}</div>
    </button>
    """

    _scripts = {
        "select": "data.clicked = true",
    }
