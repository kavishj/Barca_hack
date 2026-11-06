"""Additional physical observations; never read reference or predicted labels."""
from pathlib import Path
import numpy as np
import pandas as pd

from learned_possession.features import FeatureCache


def derivative(points, fps, span=2):
    out = np.full_like(points, np.nan, dtype=np.float32)
    if len(points) > 2 * span:
        out[span:-span] = (points[2 * span:] - points[:-2 * span]) * fps / (2 * span)
    return out


def event_history(events, fps, horizon=30):
    team = np.zeros(len(events), np.float32)
    age = np.full(len(events), np.nan, np.float32)
    previous = None
    for t, value in enumerate(events):
        if value:
            previous = t
        if previous is not None and t - previous <= horizon * fps:
            team[t], age[t] = events[previous], (t - previous) / fps
    return team, age


def known_dead(cache):
    """Supplied README match facts, independent of any training/validation file."""
    forced=np.zeros(len(cache.x),bool)
    if cache.fps==25 and len(cache.x)==199319 and cache.halves==((1,11185,90651),(2,110808,190197)):
        forced[110808:113742]=True
    return forced


def enrich(cache, match):
    from possession.signals import contacts, receptions, savgol
    from types import SimpleNamespace
    if match.n_frames != len(cache.x) or tuple(map(tuple, match.halves)) != cache.halves:
        raise ValueError('Tracking and cached feature timelines differ')
    names = {name:i for i,name in enumerate(cache.names)}
    fields = {}
    def put(name, first, stop, values):
        if name not in fields:
            fields[name] = np.full(len(cache.x), np.nan, np.float32)
        fields[name][first:stop] = values
    teams = match.team_by_frame()
    for _, first, last in cache.halves:
        stop = last + 1
        ball, players = match.ball[first:stop], match.players[first:stop]
        local = SimpleNamespace(ball=ball, players=players, fps=cache.fps)
        contact = contacts(local, teams[first:stop])
        receive = receptions(*(cache.x[first:stop, names[n]] for n in ('ball_speed', 'd_home', 'd_away')))
        for kind, events in [('contact',contact),('reception',receive)]:
            owner, age = event_history(events, cache.fps)
            put(kind + '_team_now', first, stop, events)
            put(kind + '_last_team', first, stop, owner)
            put(kind + '_age_s', first, stop, age)
            for seconds in (1,3):
                put(f'{kind}_balance_{seconds}s', first, stop,
                    pd.Series(events).rolling(seconds*cache.fps,center=True,min_periods=1).mean())
        bv, pv = derivative(ball,cache.fps), derivative(players,cache.fps)
        pv[np.linalg.norm(pv,axis=2) > 15] = np.nan  # reject reacquired slot jumps
        acceleration = savgol(ball.astype(float),2) * cache.fps**2
        put('ball_acceleration',first,stop,np.linalg.norm(acceleration,axis=1))
        distance = np.linalg.norm(players-ball[:,None],axis=2)
        for club, label in ((0,'home'),(1,'away')):
            dist = np.where((teams[first:stop]==club)&np.isfinite(distance),distance,np.inf)
            nearest = np.argmin(dist,axis=1)
            rows = np.arange(len(ball))
            relative = bv-pv[rows,nearest]
            offset = ball-players[rows,nearest]
            radius = np.linalg.norm(offset,axis=1)
            available = np.isfinite(dist[rows,nearest])
            speed = np.linalg.norm(relative,axis=1)
            radial = np.sum(relative*offset,axis=1)/np.maximum(radius,0.1)
            speed[~available], radial[~available] = np.nan,np.nan
            put(label+'_nearest_relative_speed',first,stop,speed)
            put(label+'_nearest_radial_speed',first,stop,radial)
        visible = np.isfinite(ball).all(axis=1)
        elapsed = np.zeros(len(ball),np.float32)
        for t in range(len(ball)):
            if not visible[t]:
                elapsed[t] = (elapsed[t-1] if t else 0)+1/cache.fps
        put('blackout_elapsed_s',first,stop,elapsed)
        whistle = cache.x[first:stop,names['whistle']]
        put('whistle_rise',first,stop,np.maximum(np.r_[0,np.diff(whistle)],0))
    # These supplied match facts are independent of annotation/validation files.
    # README.md defines 110808..113741 inclusive as pre-kickoff dead play.
    fields['known_dead_from_metadata'] = known_dead(cache).astype(np.float32)
    return FeatureCache(np.column_stack([cache.x,*fields.values()]),
                        (*cache.names,*fields),cache.fps,cache.halves,
                        dict(cache.provenance,structured_features=True,
                             contact_history_seconds=30,source='raw tracking; no reference/prediction labels',
                             known_dead_source='README.md supplied pre-kickoff fact; exclusive stop 113742'))


def fine_audio(wave, halves, fps=25, rate=32000):
    """20 ms spectral observations aggregated onto exact video-frame intervals.

    Broad 2–6 kHz tonality is an observation, not a referee-whistle probability.
    Frames and spectral differences never cross a graded half boundary.
    """
    if fps!=25 or rate!=32000:
        raise ValueError('Expected the timestamp-aligned 32 kHz wave and 25 fps')
    names = ('fine_audio_tonality','fine_audio_peak_hz','fine_audio_flux','fine_audio_rms','fine_audio_rise')
    result = np.full((len(wave)//(rate//fps),len(names)),np.nan,np.float32)
    hop, window, fft = rate//50, rate//25, 2048
    taper = np.hanning(window).astype(np.float32)
    hz = np.fft.rfftfreq(fft,1/rate)
    band = (hz>=2000)&(hz<=6000)
    for _,first,last in halves:
        previous = None
        previous_tone = None
        for start in range(first,last+1,750):
            stop = min(start+750,last+1)
            left, right = start*window, min((stop+1)*window,(last+1)*window)
            samples = np.asarray(wave[left:right])
            needed = (stop-start)*2*hop+window-hop
            samples = np.pad(samples,(0,max(0,needed-len(samples))))
            clips = np.lib.stride_tricks.sliding_window_view(samples,window)[::hop][:(stop-start)*2]
            spec = np.log1p(np.abs(np.fft.rfft(clips*taper,n=fft,axis=1)))[:,band]
            tone = spec.max(axis=1)-np.median(spec,axis=1)
            peak = hz[band][spec.argmax(axis=1)]
            before = np.vstack([spec[0] if previous is None else previous,spec[:-1]])
            flux = np.maximum(spec-before,0).mean(axis=1)
            rise = np.maximum(tone-np.r_[tone[0] if previous_tone is None else previous_tone,tone[:-1]],0)
            rms = np.sqrt(np.mean(clips**2,axis=1))
            rows = np.column_stack([tone,peak,flux,rms,rise]).reshape(-1,2,5)
            values = rows.mean(axis=1)
            values[:,[0,2,4]] = rows[:,:,[0,2,4]].max(axis=1)
            result[start:stop] = values
            previous, previous_tone = spec[-1],tone[-1]
    return names,result


def prepare(args):
    import games
    games.DATA_DIR = str(Path(args.data_dir).resolve())
    from possession import data
    from temporal_possession.audio import decode_audio, covered_fraction, sha
    data.CACHE_DIR = str(Path(args.out).resolve().parent/'raw_cache')
    cache = FeatureCache.load(args.features)
    rich = enrich(cache,data.load_match())
    wave_path = Path(args.out).resolve().parent/'fine_audio_wave.f32'
    print('Preparing timestamp-aligned fine audio observations',flush=True)
    wave,spans = decode_audio(args.video,wave_path,len(cache.x),cache.fps)
    names,values = fine_audio(wave,cache.halves,cache.fps)
    # Silence and absent audio are distinguishable; no fabricated values at gaps.
    available = np.zeros(len(cache.x),np.float32)
    for _,first,last in cache.halves:
        for frame in range(first,last+1):
            a,b=frame*1280,(frame+1)*1280
            available[frame]=covered_fraction(a,b,spans)
    values[available<0.95] = np.nan
    rich = FeatureCache(np.column_stack([rich.x,values,available]),
                        (*rich.names,*names,'fine_audio_available'),rich.fps,rich.halves,
                        dict(rich.provenance,base_cache_sha256=sha(args.features),
                             fine_audio_hop_ms=20,fine_audio_window_ms=40,
                             audio_timestamp_policy='PyAV resampled PTS relative to video origin'))
    rich.save(args.out)
    del wave
    wave_path.unlink()
    print(f'Structured cache: {len(rich.x)} frames x {len(rich.names)} features',flush=True)
