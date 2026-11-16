import React from 'react';
import { ArrowUpRight, FileText, Download } from 'lucide-react';
import { PageId } from '../types';

interface PublicationsPageProps {
  onNavigate: (page: PageId) => void;
}

export const PublicationsPage: React.FC<PublicationsPageProps> = () => (
  <div className="relative min-h-[75vh] w-full overflow-hidden bg-transparent text-[#15191f] dark:bg-transparent dark:text-white">
    <div className="relative mx-auto max-w-[1200px] px-5 pb-24 pt-20 sm:px-8 sm:pt-28 lg:px-14">
      <header className="mx-auto mb-16 max-w-3xl text-center">
        <h1 className="text-4xl font-semibold tracking-[-0.055em] text-[#171b22] sm:text-6xl lg:text-7xl dark:text-white">Publications<span className="text-[#a30000]">.</span></h1>
      </header>
      <section aria-label="Sferavis publication archive" className="overflow-hidden rounded-[1.5rem] border border-black/10 bg-white/60 shadow-[0_16px_50px_rgba(20,30,45,0.05)] backdrop-blur-md dark:border-white/10 dark:bg-[#0b1220]/55">
        <div className="border-b border-black/[0.07] px-6 py-6 sm:px-8 dark:border-white/10">
          <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#a30000]">Sferavis archive</p>
        </div>
        <article className="flex flex-col gap-5 px-6 py-8 sm:px-8">
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-[#a30000]/15 bg-[#a30000]/[0.06] text-[#a30000]"><FileText size={22} strokeWidth={1.5} /></div>
            <div className="min-w-0">
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#a30000]">Working paper · Draft</p>
              <h2 className="mt-2 text-xl font-semibold tracking-tight text-[#171b22] dark:text-white">R. Raian, S. Aman, I. Tsolaki, S. Tsolaki, E. Tagobert - Machine Learning based Astrometric and Spectral Classification of Newly Discovered Moving Objects in Multi-Epoch SPHEREx Imaging</h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-zinc-400">A draft manuscript describing the Sferavis project and its approach to analysing SPHEREx observations. This document is presented as a draft, not as a peer-reviewed publication.</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-4 pl-0 sm:pl-16">
            <a href={new URL('../../Sferavis_Paper_Draft.pdf', import.meta.url).href} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#A30000] hover:underline"><ArrowUpRight size={14} /> Read draft</a>
            <a href={new URL('../../Sferavis_Paper_Draft.pdf', import.meta.url).href} download="Sferavis_Paper_Draft.pdf" className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500 hover:text-[#A30000] dark:text-zinc-400"><Download size={14} /> Download PDF</a>
          </div>
        </article>
      </section>
    </div>
  </div>
);
