FROM node:22-alpine AS base

ENV PNPM_HOME="/pnpm" \
    PATH="/pnpm:$PATH"

WORKDIR /app

RUN corepack enable

FROM base AS deps

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM deps AS dev

COPY . .

EXPOSE 5173

CMD ["pnpm", "dev", "--host", "0.0.0.0", "--port", "5173"]

FROM deps AS build

COPY . .
RUN pnpm build

FROM nginx:1.27-alpine AS production

COPY nginx.conf /etc/nginx/templates/default.conf.template 
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

LABEL org.opencontainers.image.title="Psych-STRATA Dashboard frontend" \
      org.opencontainers.image.description="React frontend for the Psych-STRATA Dashboard (demo, synthetic data only)" \
      org.opencontainers.image.source="https://github.com/SCAI-BIO/psychstrata-dashboard" \
      org.opencontainers.image.licenses="Apache-2.0"

CMD ["nginx", "-g", "daemon off;"]