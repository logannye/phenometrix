import { createRemoteJWKSet, jwtVerify, SignJWT } from "jose";
import { z } from "zod";
import { invariant } from "./errors.js";

const PrincipalSchema = z.object({
  sub: z.string().min(1), tenantId: z.string().min(1),
  role: z.enum(["patient", "clinician", "researcher", "integration"]),
  subjectRefs: z.array(z.string().min(1)).min(1),
  studyIds: z.array(z.string().min(1)).min(1),
  permissions: z.array(z.enum(["episode:create", "capture:write", "consent:write", "clinical:write", "evidence:read", "evidence:review", "research:annotate"])),
  dataClass: z.enum(["synthetic", "consented-research"])
});
export type Principal = z.infer<typeof PrincipalSchema>;
export type Permission = Principal["permissions"][number];
export type Authenticate = (authorization: string | undefined) => Promise<Principal>;

export function authorize(principal: Principal, permission: Permission, subjectRef: string, dataClass: string): void {
  invariant(principal.permissions.includes(permission), 403, "permission-denied", "The session does not authorize this operation.");
  invariant(principal.subjectRefs.includes(subjectRef), 403, "subject-out-of-scope", "The participant is outside this session's scope.");
  invariant(principal.dataClass === dataClass, 403, "data-class-mismatch", "Synthetic and research sessions cannot be mixed.");
}

export function createAuthenticator(config: {
  mode: "synthetic" | "live";
  issuer: string; audience: string;
  jwksUrl?: string; developmentSecret?: Uint8Array;
}): Authenticate {
  invariant(config.issuer && config.audience, 500, "auth-config", "Issuer and audience are required.");
  if(config.mode === "live") invariant(config.jwksUrl && new URL(config.jwksUrl).protocol === "https:" && new URL(config.issuer).protocol === "https:",500,"auth-config","Live identity issuer and JWKS must use HTTPS.");
  const key = config.mode === "live"
    ? createRemoteJWKSet(new URL(config.jwksUrl!))
    : config.developmentSecret;
  invariant(key, 500, "auth-config", "A signing trust configuration is required.");
  return async (authorization) => {
    invariant(typeof authorization === "string" && authorization.startsWith("Bearer "), 401, "authentication-required", "A signed scoped session is required.");
    try {
      const options = {
        issuer: config.issuer, audience: config.audience,
        algorithms: config.mode === "live" ? ["RS256", "ES256"] : ["HS256"],
        requiredClaims: ["exp", "iat", "sub"], maxTokenAge: "1h"
      };
      const {payload}= typeof key === "function" ? await jwtVerify(authorization.slice(7),key,options) : await jwtVerify(authorization.slice(7),key!,options);
      const principal = PrincipalSchema.parse(payload);
      invariant(config.mode !== "synthetic" || principal.dataClass === "synthetic", 403, "synthetic-only", "Development sessions cannot access research data.");
      return principal;
    } catch {
      invariant(false, 401, "invalid-session", "The signed session is invalid or expired.");
    }
  };
}

/** Only wired into the explicitly synthetic loopback development launcher. */
export async function signDevelopmentSession(principal: Principal, secret: Uint8Array, issuer: string, audience: string): Promise<string> {
  invariant(principal.dataClass === "synthetic", 403, "synthetic-only", "Development sessions must be synthetic.");
  return new SignJWT(principal).setProtectedHeader({ alg: "HS256" }).setIssuedAt()
    .setExpirationTime("30m").setIssuer(issuer).setAudience(audience).sign(secret);
}
