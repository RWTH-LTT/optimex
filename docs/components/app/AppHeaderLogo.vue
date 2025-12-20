<template>
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      :srcset="darkLogo"
    >
    <img
      :src="lightLogo"
      alt="optimex logo"
      class="h-8"
    >
  </picture>
</template>

<script setup lang="ts">
const lightLogo = '/optimex_light_nomargins.svg'
const darkLogo = '/optimex_dark_nomargins.svg'
</script>
