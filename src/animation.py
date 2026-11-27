import time

from matplotlib.transforms import Bbox


class AnimationEngine:

    def __init__(
        self,
        fig,
        target_fps=60,
        text_fps=10
    ):

        self.fig = fig
        self.canvas = fig.canvas

        self.target_fps = target_fps
        self.text_interval = 1 / text_fps

        self.artists = []
        self.regions = []

        self.last_text_update = 0.0
        self.last_fps_update = time.perf_counter()

        self.frame_count = 0
        self.fps = 0.0

        self.full_draws = 0
        self.blit_draws = 0

        self.render_time = 0.0

        self.closed = False
        self.supports_blit = self.canvas.supports_blit

        self.canvas.mpl_connect(
            "draw_event",
            self.on_draw
        )

        self.canvas.mpl_connect(
            "resize_event",
            self.invalidate
        )

        self.canvas.mpl_connect(
            "close_event",
            self.on_close
        )

    def on_close(self, event):

        self.closed = True
        self.regions = []

    def invalidate(self, event=None):

        self.regions = []

    def set_artists(self, *artists):

        for artist in self.artists:

            artist.set_animated(False)

        self.artists = list(artists)

        for artist in self.artists:

            artist.set_animated(
                self.supports_blit
            )

        self.invalidate()

    def get_region(self):

        bbox = self.fig.bbox

        return Bbox.from_extents(
            bbox.x0,
            bbox.y0 + bbox.height * 0.30,
            bbox.x1,
            bbox.y1
        )

    def build_regions(self):

        groups = {}

        for artist in self.artists:

            if artist.axes is not None:

                key = artist.axes

            else:

                key = self.fig

            groups.setdefault(
                key,
                []
            ).append(artist)

        regions = []

        for owner, artists in groups.items():

            if owner is self.fig:

                bbox = self.fig.bbox

                region = Bbox.from_extents(
                    bbox.x0 + bbox.width * 0.69,
                    bbox.y0 + bbox.height * 0.30,
                    bbox.x1,
                    bbox.y0 + bbox.height * 0.67
                )

            else:

                region = owner.bbox

            regions.append({
                "bbox": region,
                "artists": artists,
                "background": self.canvas.copy_from_bbox(
                    region
                )
            })

        return regions

    def draw_artists(self, renderer, artists):

        for artist in sorted(
            artists,
            key=lambda item: item.get_zorder()
        ):

            if artist.get_visible():

                artist.draw(renderer)

    def on_draw(self, event):

        if self.closed or not self.supports_blit:
            return

        if event.canvas is not self.canvas:
            return

        self.regions = self.build_regions()

        for region in self.regions:

            self.draw_artists(
                event.renderer,
                region["artists"]
            )

        self.full_draws += 1

    def should_update_text(self, force=False):

        now = time.perf_counter()

        if force or (
            now - self.last_text_update
            >= self.text_interval
        ):

            self.last_text_update = now

            return True

        return False

    def update_fps(self):

        self.frame_count += 1

        now = time.perf_counter()

        elapsed = now - self.last_fps_update

        if elapsed >= 1.0:

            self.fps = self.frame_count / elapsed

            self.frame_count = 0
            self.last_fps_update = now

    def render(self, update_text=True):

        if self.closed:
            return

        start = time.perf_counter()

        if not self.supports_blit:

            self.canvas.draw_idle()

        elif not self.regions:

            self.canvas.draw()

        else:

            renderer = self.canvas.get_renderer()

            for region in self.regions:

                is_text_region = all(
                    artist.axes is None
                    for artist in region["artists"]
                )

                if is_text_region and not update_text:
                    continue

                self.canvas.restore_region(
                    region["background"]
                )

                self.draw_artists(
                    renderer,
                    region["artists"]
                )

                self.canvas.blit(
                    region["bbox"]
                )

            self.blit_draws += 1

        elapsed = time.perf_counter() - start

        if self.render_time == 0:

            self.render_time = elapsed

        else:

            self.render_time = (
                0.9 * self.render_time
                + 0.1 * elapsed
            )

        self.update_fps()

    def reset_fps(self):

        self.frame_count = 0
        self.fps = 0.0

        self.full_draws = 0
        self.blit_draws = 0

        self.render_time = 0.0

        self.last_fps_update = time.perf_counter()

    def get_fps(self):

        return self.fps

    def get_render_time(self):

        return self.render_time * 1000

    def get_timer_interval(self):

        return max(
            1,
            round(1000 / self.target_fps)
        )

    def get_statistics(self):

        return {
            "fps": self.fps,
            "full_draws": self.full_draws,
            "blit_draws": self.blit_draws,
            "render_time_ms": self.get_render_time()
        }