use axum::{
    extract::Multipart,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use serde_json::json;
use std::net::SocketAddr;
use std::process::Command;
use tower_http::cors::CorsLayer;

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    version: String,
    engine: String,
}

async fn health_check() -> impl IntoResponse {
    Json(StatusResponse {
        status: "online".to_string(),
        version: "0.2.0".to_string(),
        engine: "Nexus-Forge Fusion (HoVer-Net + geometry)".to_string(),
    })
}

/// Receive an image (multipart field `file`), run the offline fusion pipeline
/// (Python CLI), and return the per-cell malignancy JSON.
async fn analyze_image(mut multipart: Multipart) -> impl IntoResponse {
    let mut bytes: Vec<u8> = Vec::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name().unwrap_or("") == "file" {
            if let Ok(data) = field.bytes().await {
                bytes = data.to_vec();
            }
        }
    }

    if bytes.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"status": "error", "message": "empty file"})));
    }

    // write to a temp file
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_path = std::env::temp_dir().join(format!(
        "nexus_{}_{}.png",
        std::process::id(),
        nanos
    ));
    if let Err(e) = std::fs::write(&tmp_path, &bytes) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"status": "error", "message": format!("write temp: {e}")})),
        );
    }

    // locate the Python CLI (env override for deployment)
    let default_cli = std::env::current_exe()
        .ok()
        .and_then(|p| {
            let mut root = p.parent()?.parent()?.parent()?.to_path_buf();
            root.push("nexus-forge-cyto-core/python/nexus_predict_cli.py");
            Some(root)
        })
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "nexus-forge-cyto-core/python/nexus_predict_cli.py".to_string());
    let cli_path = std::env::var("NEXUS_CLI_PATH").unwrap_or(default_cli);
    let python = std::env::var("NEXUS_PYTHON").unwrap_or_else(|_| "python".to_string());

    let result = Command::new(&python)
        .arg(&cli_path)
        .arg(&tmp_path)
        .output();

    let _ = std::fs::remove_file(&tmp_path);

    match result {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) => (StatusCode::OK, Json(v)),
                Err(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"status": "error", "message": format!("bad json: {text}")})),
                ),
            }
        }
        Ok(out) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"status": "error", "message": String::from_utf8_lossy(&out.stderr).trim().to_string()})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"status": "error", "message": format!("python failed: {e}")})),
        ),
    }
}

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::AllowOrigin::any())
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/api/health", get(health_check))
        .route("/analyze-image", post(analyze_image))
        .layer(cors)
        .layer(axum::middleware::from_fn(security_headers));

    async fn security_headers(
        response: axum::response::Response,
    ) -> axum::response::Response {
        let mut resp = response;
        let headers = resp.headers_mut();
        headers.insert(
            axum::http::header::HeaderName::from_static("x-content-type-options"),
            axum::http::header::HeaderValue::from_static("nosniff"),
        );
        headers.insert(
            axum::http::header::HeaderName::from_static("x-frame-options"),
            axum::http::header::HeaderValue::from_static("DENY"),
        );
        resp
    }

    let addr = SocketAddr::from(([0, 0, 0, 0], 8811));
    println!("Nexus-Viewer API server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
