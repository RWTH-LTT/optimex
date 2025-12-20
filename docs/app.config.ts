export default defineAppConfig({
  docus: {
    title: 'optimex',
    description: 'Time-explicit transition pathway optimization based on Life Cycle Assessment (LCA)',
    image: 'https://raw.githubusercontent.com/RWTH-LTT/optimex/main/docs-old/_static/optimex_preview.png',
    socials: {
      github: 'RWTH-LTT/optimex'
    },
    aside: {
      level: 0,
      exclude: []
    },
    header: {
      logo: {
        light: '/optimex_light_nomargins.svg',
        dark: '/optimex_dark_nomargins.svg'
      },
      title: false
    },
    footer: {
      credits: 'Copyright © 2025 optimex developers',
      iconLinks: [
        {
          href: 'https://github.com/RWTH-LTT/optimex',
          icon: 'simple-icons:github',
          label: 'optimex on GitHub'
        }
      ]
    }
  }
})
