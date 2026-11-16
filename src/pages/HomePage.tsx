import React from 'react';
import { PageId } from '../types';
import { ArrowRight, ChevronRight } from 'lucide-react';

const spectrumGallery = [
  {
    src: '/images/home-spectrum/spherex-light-curve.jpg',
    alt: 'SPHEREx spectrum and light curve plots showing flux across wavelength and observation time.',
    caption: 'SPHEREx spectrum and light curve',
  },
  {
    src: '/images/home-spectrum/psf-spectrum.jpg',
    alt: 'Binned and filtered SPHEREx point-spread-function spectrum for T2 SIMP J013656.5+093347.3.',
    caption: 'Point-spread-function spectrum',
  },
  {
    src: '/images/home-spectrum/spectrum-and-classification.jpg',
    alt: 'Combined spectrum, wavelength-coloured observation measurements, and classification scores for a target.',
    caption: 'Spectrum, observations, and classification',
  },
  {
    src: '/images/home-spectrum/spectrum-standard-comparison.jpg',
    alt: 'Three target spectra compared with standards across near-infrared wavelengths.',
    caption: 'Comparison with spectral standards',
  },
  {
    src: '/images/home-spectrum/aperture-spectrum.jpg',
    alt: 'Binned and filtered SPHEREx aperture spectrum for T2 SIMP J013656.5+093347.3.',
    caption: 'Aperture spectrum',
  },
  {
    src: '/images/home-spectrum/brown-dwarf-spectra.jpg',
    alt: 'Infrared spectra of L and T dwarf objects plotted across wavelengths from one to five micrometres.',
    caption: 'L and T dwarf spectra',
  },
  {
    src: '/images/home-spectrum/sferavis-pipeline.jfif',
    alt: 'Sferavis pipeline diagram from SPHEREx image acquisition through source detection, tracklet linking, and spectral classification.',
    caption: 'The Sferavis data pipeline',
  },
];

interface HomePageProps {
  onNavigate: (page: PageId) => void;
  onLaunchSferavis: () => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate, onLaunchSferavis }) => {
  return (
    <main className="relative w-full overflow-hidden px-4 py-16 text-center text-slate-900 dark:text-white sm:py-20">
      <div
        className="pointer-events-none absolute left-1/2 top-1/3 h-[28rem] w-[42rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#A30000]/10 blur-[140px]"
        aria-hidden="true"
      />
      <div className="relative z-10 mx-auto w-full max-w-6xl">
        <h1 className="mx-auto max-w-5xl text-4xl font-oswald font-black uppercase leading-[0.98] tracking-tight text-slate-950 dark:text-white text-balance sm:text-6xl lg:text-8xl">
          Exploring the{' '}
          <span className="text-[#A30000]">Infrared Universe.</span>
        </h1>

        <p className="mx-auto mt-6 max-w-3xl text-base leading-relaxed text-slate-700 dark:text-zinc-300 text-balance sm:text-xl">
          An independent science platform exploring the SPHEREx mission, its near-infrared survey, and the astrophysics behind cosmic inflation, galaxy formation, and the origins of interstellar ices.
        </p>

        <section className="mt-12 text-left" aria-labelledby="home-gallery-heading">
          <h2
            id="home-gallery-heading"
            className="mb-5 text-center font-oswald text-2xl font-bold uppercase tracking-wide text-slate-900 dark:text-white sm:text-3xl"
          >
            SPHEREx Research Highlights
          </h2>
          <div className="columns-1 gap-5 sm:columns-2 xl:columns-3">
            {spectrumGallery.map((image) => (
              <figure
                key={image.src}
                className="mb-5 break-inside-avoid overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-950/80"
              >
                <img
                  src={image.src}
                  alt={image.alt}
                  loading="lazy"
                  decoding="async"
                  className="block h-auto w-full"
                />
                <figcaption className="border-t border-slate-200 px-4 py-3 text-sm font-medium text-slate-700 dark:border-slate-700 dark:text-slate-200">
                  {image.caption}
                </figcaption>
              </figure>
            ))}
          </div>
        </section>

        <section className="mt-12" aria-labelledby="home-video-heading">
          <h2
            id="home-video-heading"
            className="mb-5 text-center font-oswald text-2xl font-bold uppercase tracking-wide text-slate-900 dark:text-white sm:text-3xl"
          >
            Sferavis in Action
          </h2>
          <video
            className="mx-auto block h-auto max-h-[70vh] w-full rounded-xl border border-slate-200 bg-black shadow-sm dark:border-slate-700"
            controls
            playsInline
            preload="metadata"
            aria-label="Video showcasing Sferavis"
          >
            <source src="/videos/sferavis-homepage.mp4" type="video/mp4" />
            Your browser does not support embedded video.
          </video>
        </section>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <button
            type="button"
            onClick={onLaunchSferavis}
            className="inline-flex items-center gap-2 bg-[#A30000] px-8 py-4 text-xs font-lexend font-bold uppercase tracking-widest text-white transition-colors hover:bg-[#8B0000] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#A30000] focus-visible:ring-offset-2"
          >
            Launch Sferavis <ArrowRight className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onNavigate('spherex')}
            className="inline-flex items-center gap-2 border border-zinc-300 bg-white px-8 py-4 text-xs font-lexend font-semibold uppercase tracking-wider text-slate-800 transition-colors hover:border-[#A30000] hover:text-[#A30000] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#A30000] dark:border-slate-700 dark:bg-slate-900 dark:text-zinc-200 dark:hover:border-[#A30000] dark:hover:text-white"
          >
            The SPHEREx Mission <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </main>
  );
};
